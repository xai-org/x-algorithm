use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use xai_abuse_proto::enforcement::ScoreResult;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    #[default]
    User,
    Post,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::User => "user",
            EntityType::Post => "post",
        }
    }

    pub fn from_path(s: &str) -> Option<Self> {
        match s {
            "user" => Some(EntityType::User),
            "post" => Some(EntityType::Post),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Facts {
    #[serde(default)]
    pub entity_type: EntityType,
    #[serde(default)]
    pub entity_id: i64,
    pub user_id: i64,
    pub topic: String,
    pub score: ScoreFacts,
    pub entity: EntityFacts,
    pub uas: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UasClientActivity {
    pub client_app_id: String,
    pub active_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UasBody {
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub last_day: Vec<UasClientActivity>,
    #[serde(default)]
    pub last_week: Vec<UasClientActivity>,
    #[serde(default)]
    pub last_month: Vec<UasClientActivity>,
    #[serde(default)]
    pub last_year: Vec<UasClientActivity>,
}

impl UasBody {
    pub fn parse(raw: &str) -> Option<Self> {
        serde_json::from_str(raw).ok()
    }

    pub fn activities(&self) -> impl Iterator<Item = (&'static str, &UasClientActivity)> {
        self.last_day
            .iter()
            .map(|a| ("lastDay", a))
            .chain(self.last_week.iter().map(|a| ("lastWeek", a)))
            .chain(self.last_month.iter().map(|a| ("lastMonth", a)))
            .chain(self.last_year.iter().map(|a| ("lastYear", a)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityFacts {
    User(UserFacts),
    Post(PostFacts),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserFacts {
    #[serde(default)]
    pub allowlist: AllowlistFacts,
    pub user: GizmoduckFacts,
    pub cred: CredFacts,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PostFacts {
    #[serde(default)]
    pub allowlist: AllowlistFacts,
    pub present: bool,
    pub author_id: i64,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub author: UserFacts,
    #[serde(default)]
    pub invited: InvitedReply,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvitedReply {
    #[default]
    NotAReply,
    Mentioned,
    NotInvited,
    Unknown,
}

impl InvitedReply {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotAReply => "not_a_reply",
            Self::Mentioned => "mentioned",
            Self::NotInvited => "not_invited",
            Self::Unknown => "unknown",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "mentioned" => Self::Mentioned,
            "not_invited" => Self::NotInvited,
            "unknown" => Self::Unknown,
            _ => Self::NotAReply,
        }
    }
}

impl Facts {
    pub fn user_allowlist(&self) -> &AllowlistFacts {
        &self.user_signals().allowlist
    }
    pub fn user(&self) -> &GizmoduckFacts {
        &self.user_signals().user
    }
    pub fn cred(&self) -> &CredFacts {
        &self.user_signals().cred
    }
    pub fn user_allowlist_mut(&mut self) -> &mut AllowlistFacts {
        &mut self.user_signals_mut().allowlist
    }
    pub fn user_mut(&mut self) -> &mut GizmoduckFacts {
        &mut self.user_signals_mut().user
    }
    pub fn cred_mut(&mut self) -> &mut CredFacts {
        &mut self.user_signals_mut().cred
    }

    fn user_signals(&self) -> &UserFacts {
        match &self.entity {
            EntityFacts::User(s) => s,
            EntityFacts::Post(p) => &p.author,
        }
    }
    fn user_signals_mut(&mut self) -> &mut UserFacts {
        match &mut self.entity {
            EntityFacts::User(s) => s,
            EntityFacts::Post(p) => &mut p.author,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreFacts {
    pub model_version: String,
    pub enforcement_note: String,
    pub decoded_actions: String,
    pub labels: Vec<String>,
    #[serde(default)]
    pub skip_author_credibility_prechecks: bool,
}

impl ScoreFacts {
    pub fn from_score(score: &ScoreResult) -> Self {
        let v = serde_json::to_value(score).unwrap_or_default();
        let take_string = |key: &str| -> String {
            match v.get(key) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            }
        };
        let labels = score
            .summary
            .as_ref()
            .map(|s| {
                s.labels
                    .iter()
                    .map(|l| crate::action_control::strip_dry_run_prefix(l).to_owned())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            model_version: take_string("model_version"),
            enforcement_note: take_string("enforcement_note"),
            decoded_actions: take_string("decoded_actions"),
            labels,
            skip_author_credibility_prechecks: score.skip_author_credibility_prechecks,
        }
    }
}

pub fn head_label(score: &ScoreResult) -> &str {
    score
        .summary
        .as_ref()
        .and_then(|s| s.fired_heads.first())
        .map(|h| h.name.as_str())
        .unwrap_or_default()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AllowlistFacts {
    pub is_allowlisted: bool,
    pub added_by: Option<String>,
    pub reason: Option<String>,
    pub ttl_secs: Option<i64>,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct GizmoduckFacts {
    pub present: bool,
    pub suspended: bool,
    pub deactivated: bool,
    #[serde(default)]
    pub labels: Vec<String>,
    pub is_blue_verified: bool,
    pub verified_type: Option<String>,
    pub is_affiliate: bool,
    pub user_type: Option<String>,
    #[serde(default)]
    pub test_user_status: Option<String>,
    #[serde(default, skip_serializing)]
    pub roles: Vec<String>,
    #[serde(default, skip_serializing)]
    pub login_email: Option<String>,
    #[serde(default)]
    pub email_confirmed: Option<EmailConfirmedState>,
    #[serde(default, skip_serializing)]
    pub creation_ip: Option<String>,
    #[serde(default)]
    pub test_signal: Option<crate::safety_checks::TestAccountSignal>,
    #[serde(default)]
    pub permanently_suspended: bool,
}

impl std::fmt::Debug for GizmoduckFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let set = |v: bool| if v { "<set>" } else { "<none>" };
        f.debug_struct("GizmoduckFacts")
            .field("present", &self.present)
            .field("suspended", &self.suspended)
            .field("deactivated", &self.deactivated)
            .field("labels", &self.labels)
            .field("is_blue_verified", &self.is_blue_verified)
            .field("verified_type", &self.verified_type)
            .field("is_affiliate", &self.is_affiliate)
            .field("user_type", &self.user_type)
            .field("test_user_status", &self.test_user_status)
            .field("roles", &self.roles.len())
            .field("login_email", &set(self.login_email.is_some()))
            .field("email_confirmed", &self.email_confirmed)
            .field("creation_ip", &set(self.creation_ip.is_some()))
            .field("test_signal", &self.test_signal)
            .field("permanently_suspended", &self.permanently_suspended)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmailConfirmedState {
    Grandfathered,
    Excused,
    Unconfirmed,
    Confirmed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CredFacts {
    pub is_high: Option<bool>,
    pub verified_type: Option<String>,
    pub score: Option<f64>,
    pub follower_count: Option<i64>,
}

impl Facts {
    pub fn resolve_target(score: &ScoreResult) -> (EntityType, i64) {
        use xai_abuse_proto::enforcement::EntityType as ProtoEntityType;
        let proto_type =
            ProtoEntityType::try_from(score.entity_type).unwrap_or(ProtoEntityType::Unspecified);
        match proto_type {
            ProtoEntityType::Post => (EntityType::Post, score.entity_id),
            ProtoEntityType::User | ProtoEntityType::Unspecified => {
                let id = if score.entity_id != 0 {
                    score.entity_id
                } else {
                    score.user_id
                };
                (EntityType::User, id)
            }
        }
    }

    pub fn to_additional_info_map(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();

        m.insert("entity_type".into(), self.entity_type.as_str().into());
        m.insert("entity_id".into(), self.entity_id.to_string());

        m.insert("decoded_actions".into(), self.score.decoded_actions.clone());
        m.insert(
            "enforcement_note".into(),
            self.score.enforcement_note.clone(),
        );
        m.insert("model_version".into(), self.score.model_version.clone());
        m.insert(
            "skip_author_credibility_prechecks".into(),
            self.score.skip_author_credibility_prechecks.to_string(),
        );
        m.insert("source_topic".into(), self.topic.clone());

        match &self.entity {
            EntityFacts::User(s) => emit_user_signals(&mut m, s),
            EntityFacts::Post(p) => emit_post_facts(&mut m, p),
        }

        if let Some(resp) = &self.uas {
            m.insert("uas_response".into(), resp.clone());
        }

        m
    }
}

fn emit_allowlist_block(m: &mut BTreeMap<String, String>, prefix: &str, al: &AllowlistFacts) {
    if !al.is_allowlisted {
        return;
    }
    m.insert(format!("{prefix}_is_allowlisted"), "true".into());
    m.insert(
        format!("{prefix}_added_by"),
        fmt_opt_dbg(al.added_by.as_deref()),
    );
    m.insert(
        format!("{prefix}_reason"),
        fmt_opt_dbg(al.reason.as_deref()),
    );
    m.insert(
        format!("{prefix}_ttl_secs"),
        fmt_opt_dbg(al.ttl_secs.as_ref()),
    );
}

fn parse_allowlist_block(map: &BTreeMap<String, String>, prefix: &str) -> AllowlistFacts {
    if !parse_bool(map.get(&format!("{prefix}_is_allowlisted"))) {
        return AllowlistFacts::default();
    }
    AllowlistFacts {
        is_allowlisted: true,
        added_by: parse_dbg_opt_string(map.get(&format!("{prefix}_added_by"))),
        reason: parse_dbg_opt_string(map.get(&format!("{prefix}_reason"))),
        ttl_secs: parse_dbg_opt_i64(map.get(&format!("{prefix}_ttl_secs"))),
    }
}

fn emit_user_signals(m: &mut BTreeMap<String, String>, s: &UserFacts) {
    emit_allowlist_block(m, "allowlist", &s.allowlist);

    if s.user.present {
        m.insert("gizmoduck_suspended".into(), s.user.suspended.to_string());
        m.insert(
            "gizmoduck_deactivated".into(),
            s.user.deactivated.to_string(),
        );
        m.insert(
            "gizmoduck_labels".into(),
            serde_json::to_string(&s.user.labels).unwrap_or_else(|_| "[]".into()),
        );
        m.insert(
            "gizmoduck_is_blue_verified".into(),
            s.user.is_blue_verified.to_string(),
        );
        m.insert(
            "gizmoduck_verified_type".into(),
            fmt_opt_dbg(s.user.verified_type.as_deref()),
        );
        m.insert(
            "gizmoduck_is_affiliate".into(),
            s.user.is_affiliate.to_string(),
        );
        if s.user.permanently_suspended {
            m.insert("gizmoduck_permanently_suspended".into(), "true".into());
        }
    }

    m.insert("cred_is_high".into(), fmt_opt_dbg(s.cred.is_high.as_ref()));
    m.insert(
        "cred_verified_type".into(),
        fmt_opt_dbg(s.cred.verified_type.as_deref()),
    );
    m.insert("cred_score".into(), fmt_opt_dbg(s.cred.score.as_ref()));
    m.insert(
        "cred_follower_count".into(),
        fmt_opt_dbg(s.cred.follower_count.as_ref()),
    );
}

fn emit_post_facts(m: &mut BTreeMap<String, String>, p: &PostFacts) {
    m.insert("post_present".into(), p.present.to_string());
    m.insert("post_author_id".into(), p.author_id.to_string());
    m.insert(
        "post_labels".into(),
        serde_json::to_string(&p.labels).unwrap_or_else(|_| "[]".into()),
    );
    emit_allowlist_block(m, "post_allowlist", &p.allowlist);
    if p.invited != InvitedReply::NotAReply {
        m.insert("post_invited_reply".into(), p.invited.as_str().into());
    }
    emit_user_signals(m, &p.author);
}

impl Facts {
    pub fn from_additional_info_map(
        map: &BTreeMap<String, String>,
        score: &ScoreResult,
        topic: &str,
    ) -> Option<Self> {
        if !map.contains_key("cred_is_high") && !map.contains_key("entity_type") {
            return None;
        }

        let (default_type, default_id) = Facts::resolve_target(score);
        let entity_type = match map.get("entity_type").map(String::as_str) {
            Some("post") => EntityType::Post,
            Some("user") => EntityType::User,
            _ => default_type,
        };
        let entity_id = map
            .get("entity_id")
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|id| *id != 0)
            .unwrap_or(default_id);

        let entity = match entity_type {
            EntityType::User => EntityFacts::User(reconstruct_user_signals(map)),
            EntityType::Post => EntityFacts::Post(PostFacts {
                present: parse_bool(map.get("post_present")),
                author_id: map
                    .get("post_author_id")
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(score.user_id),
                labels: map
                    .get("post_labels")
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default(),
                allowlist: parse_allowlist_block(map, "post_allowlist"),
                author: reconstruct_user_signals(map),
                invited: map
                    .get("post_invited_reply")
                    .map_or(InvitedReply::NotAReply, |s| InvitedReply::parse(s)),
            }),
        };

        let user_id = match entity_type {
            EntityType::User => entity_id,
            EntityType::Post => score.user_id,
        };

        Some(Facts {
            entity_type,
            entity_id,
            user_id,
            topic: topic.to_owned(),
            score: ScoreFacts::from_score(score),
            entity,
            uas: map.get("uas_response").cloned(),
        })
    }
}

fn reconstruct_user_signals(map: &BTreeMap<String, String>) -> UserFacts {
    let allowlist = parse_allowlist_block(map, "allowlist");

    let user = if map.contains_key("gizmoduck_suspended") {
        GizmoduckFacts {
            present: true,
            suspended: parse_bool(map.get("gizmoduck_suspended")),
            deactivated: parse_bool(map.get("gizmoduck_deactivated")),
            labels: parse_user_labels(map),
            is_blue_verified: parse_bool(map.get("gizmoduck_is_blue_verified")),
            verified_type: parse_dbg_opt_string(map.get("gizmoduck_verified_type")),
            is_affiliate: parse_bool(map.get("gizmoduck_is_affiliate")),
            permanently_suspended: parse_bool(map.get("gizmoduck_permanently_suspended")),
            ..Default::default()
        }
    } else {
        GizmoduckFacts::default()
    };

    let cred = CredFacts {
        is_high: parse_dbg_opt_bool(map.get("cred_is_high")),
        verified_type: parse_dbg_opt_string(map.get("cred_verified_type")),
        score: parse_dbg_opt_f64(map.get("cred_score")),
        follower_count: parse_dbg_opt_i64(map.get("cred_follower_count")),
    };

    UserFacts {
        allowlist,
        user,
        cred,
    }
}

fn fmt_opt_dbg<T: std::fmt::Debug + ?Sized>(v: Option<&T>) -> String {
    format!("{v:?}")
}

fn parse_bool(v: Option<&String>) -> bool {
    v.map(String::as_str) == Some("true")
}

fn parse_dbg_opt_bool(v: Option<&String>) -> Option<bool> {
    match v.map(String::as_str)? {
        "Some(true)" => Some(true),
        "Some(false)" => Some(false),
        _ => None,
    }
}

fn parse_dbg_opt_i64(v: Option<&String>) -> Option<i64> {
    let s = v.map(String::as_str)?;
    s.strip_prefix("Some(")?.strip_suffix(')')?.parse().ok()
}

fn parse_dbg_opt_f64(v: Option<&String>) -> Option<f64> {
    let s = v.map(String::as_str)?;
    s.strip_prefix("Some(")?.strip_suffix(')')?.parse().ok()
}

fn parse_dbg_opt_string(v: Option<&String>) -> Option<String> {
    let s = v.map(String::as_str)?;
    s.strip_prefix("Some(\"")?
        .strip_suffix("\")")
        .map(str::to_owned)
}

fn parse_user_labels(map: &BTreeMap<String, String>) -> Vec<String> {
    if let Some(json) = map.get("gizmoduck_labels") {
        return serde_json::from_str(json).unwrap_or_default();
    }
    if parse_bool(map.get("gizmoduck_has_spam_high_recall")) {
        return vec!["SpamHighRecall".to_owned()];
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_abuse_proto::enforcement::{ScoreResult, SummaryCounters};

    fn score_with_labels(labels: &[&str]) -> ScoreResult {
        ScoreResult {
            summary: Some(SummaryCounters {
                labels: labels.iter().map(|l| (*l).into()).collect(),
                ..Default::default()
            }),
            model_version: "v_test".into(),
            enforcement_note: "test note".into(),
            ..Default::default()
        }
    }

    #[test]
    fn score_facts_flattens_labels_into_a_vec() {
        let s = score_with_labels(&["RTBot", "content_spammer"]);
        let f = ScoreFacts::from_score(&s);
        assert_eq!(f.labels, vec!["RTBot".to_owned(), "content_spammer".into()]);
        assert_eq!(f.model_version, "v_test");
        assert_eq!(f.enforcement_note, "test note");
    }

    #[test]
    fn score_facts_labels_empty_when_summary_absent() {
        let s = ScoreResult {
            summary: None,
            ..Default::default()
        };
        let f = ScoreFacts::from_score(&s);
        assert!(f.labels.is_empty());
    }

    #[test]
    fn account_details_never_leave_the_lookup() {
        let plain = GizmoduckFacts {
            present: true,
            labels: vec!["SpamHighRecall".into()],
            ..Default::default()
        };
        let detailed = GizmoduckFacts {
            test_user_status: Some("Temporary".into()),
            roles: vec!["qa_test_role".into()],
            login_email: Some("someone@test.example.com".into()),
            email_confirmed: Some(EmailConfirmedState::Confirmed),
            creation_ip: Some("203.0.113.9".into()),
            test_signal: Some(crate::safety_checks::TestAccountSignal::EmailDomain),
            ..plain.clone()
        };
        let facts = |entity: EntityType, user: &GizmoduckFacts| {
            let account = UserFacts {
                user: user.clone(),
                ..Default::default()
            };
            Facts {
                entity_type: entity,
                entity_id: 7,
                user_id: 7,
                topic: "t".into(),
                score: ScoreFacts::from_score(&ScoreResult::default()),
                entity: match entity {
                    EntityType::User => EntityFacts::User(account),
                    EntityType::Post => EntityFacts::Post(PostFacts {
                        author_id: 7,
                        author: account,
                        ..Default::default()
                    }),
                },
                uas: None,
            }
        };
        for entity in [EntityType::User, EntityType::Post] {
            let with = facts(entity, &detailed);
            assert_eq!(
                with.to_additional_info_map(),
                facts(entity, &plain).to_additional_info_map(),
                "{entity:?}"
            );
            let shown = format!(
                "{:?} {} {:?}",
                with.to_additional_info_map(),
                serde_json::to_string(&detailed).unwrap(),
                with.user()
            );
            for secret in ["someone", "test.example.com", "203.0.113", "qa_test_role"] {
                assert!(!shown.contains(secret), "{secret} in {shown}");
            }
        }
        let old: GizmoduckFacts =
            serde_json::from_str(r#"{"present":true,"suspended":false,"deactivated":false,"labels":[],"is_blue_verified":false,"verified_type":null,"is_affiliate":false,"user_type":null}"#)
                .unwrap();
        assert!(old.roles.is_empty() && old.login_email.is_none() && old.test_signal.is_none());
    }

    #[test]
    fn audit_map_omits_gizmoduck_keys_when_user_absent() {
        let facts = Facts {
            entity_type: EntityType::User,
            entity_id: 1,
            user_id: 1,
            topic: "t".into(),
            score: ScoreFacts::from_score(&ScoreResult::default()),
            entity: EntityFacts::User(UserFacts {
                allowlist: AllowlistFacts::default(),
                user: GizmoduckFacts::default(),
                cred: CredFacts::default(),
            }),
            uas: None,
        };
        let m = facts.to_additional_info_map();
        assert!(!m.contains_key("gizmoduck_suspended"));
        assert_eq!(m.get("source_topic").map(String::as_str), Some("t"));
        assert!(!m.contains_key("ads_account_status"));
    }

    #[test]
    fn from_additional_info_map_returns_none_for_legacy_empty_map() {
        let score = ScoreResult::default();
        assert!(Facts::from_additional_info_map(&BTreeMap::new(), &score, "t").is_none());

        let mut map = BTreeMap::new();
        map.insert("model_version".into(), "v1".into());
        map.insert("source_topic".into(), "t".into());
        assert!(Facts::from_additional_info_map(&map, &score, "t").is_none());
    }

    #[test]
    fn from_additional_info_map_reconstructs_user_in_allowlist_skip() {
        let score = ScoreResult {
            user_id: 42,
            model_version: "v_test".into(),
            ..Default::default()
        };
        let original = Facts {
            entity_type: EntityType::User,
            entity_id: 42,
            user_id: 42,
            topic: "t".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::User(UserFacts {
                allowlist: AllowlistFacts {
                    is_allowlisted: true,
                    added_by: Some("oncall".into()),
                    reason: Some("vip account".into()),
                    ttl_secs: Some(86_400),
                },
                ..Default::default()
            }),
            uas: None,
        };
        let map = original.to_additional_info_map();
        assert_eq!(map["allowlist_is_allowlisted"], "true");
        assert_eq!(map["allowlist_added_by"], r#"Some("oncall")"#);
        assert_eq!(map["allowlist_reason"], r#"Some("vip account")"#);
        assert_eq!(map["allowlist_ttl_secs"], "Some(86400)");
        let parsed = Facts::from_additional_info_map(&map, &score, "t")
            .expect("user_in_allowlist skip-path map should reconstruct");

        assert!(parsed.user_allowlist().is_allowlisted);
        assert_eq!(parsed.user_allowlist().added_by.as_deref(), Some("oncall"));
        assert_eq!(
            parsed.user_allowlist().reason.as_deref(),
            Some("vip account")
        );
        assert_eq!(parsed.user_allowlist().ttl_secs, Some(86_400));
        assert!(!parsed.user().present);
        assert_eq!(parsed.user_id, 42);
    }

    #[test]
    fn from_additional_info_map_ignores_retired_ads_keys() {
        let score = ScoreResult::default();
        let mut legacy = BTreeMap::new();
        legacy.insert("model_version".into(), "v_test".into());
        legacy.insert("source_topic".into(), "t".into());
        legacy.insert("cred_is_high".into(), "Some(false)".into());
        legacy.insert("cred_verified_type".into(), "None".into());
        legacy.insert("cred_score".into(), "None".into());
        legacy.insert("cred_follower_count".into(), "Some(0)".into());
        legacy.insert("ads_account_dormant_status".into(), "Active".into());
        legacy.insert("ads_max_active_spend_micro".into(), "1500000".into());
        legacy.insert("ads_account_id".into(), "42".into());
        legacy.insert("ads_account_legacy_user_id".into(), "100".into());
        legacy.insert("ads_account_json".into(), "[]".into());
        legacy.insert("gizmoduck_suspended".into(), "false".into());
        legacy.insert("gizmoduck_deactivated".into(), "false".into());
        legacy.insert("gizmoduck_labels".into(), "[]".into());
        legacy.insert("gizmoduck_is_blue_verified".into(), "false".into());
        legacy.insert("gizmoduck_verified_type".into(), "None".into());
        legacy.insert("gizmoduck_is_affiliate".into(), "false".into());

        let parsed = Facts::from_additional_info_map(&legacy, &score, "t")
            .expect("maps carrying retired ads keys must still reconstruct");
        assert!(parsed.user().present);
        assert_eq!(parsed.cred().is_high, Some(false));
    }

    #[test]
    fn from_additional_info_map_falls_back_to_legacy_has_spam_high_recall_bool() {
        let score = ScoreResult {
            user_id: 7,
            ..Default::default()
        };
        let mut legacy = BTreeMap::new();
        legacy.insert("model_version".into(), "v_test".into());
        legacy.insert("source_topic".into(), "t".into());
        legacy.insert("cred_is_high".into(), "Some(false)".into());
        legacy.insert("cred_verified_type".into(), "None".into());
        legacy.insert("cred_score".into(), "None".into());
        legacy.insert("cred_follower_count".into(), "Some(0)".into());
        legacy.insert("gizmoduck_suspended".into(), "false".into());
        legacy.insert("gizmoduck_deactivated".into(), "false".into());
        legacy.insert("gizmoduck_has_spam_high_recall".into(), "true".into());
        legacy.insert("gizmoduck_is_blue_verified".into(), "false".into());
        legacy.insert("gizmoduck_verified_type".into(), "None".into());
        legacy.insert("gizmoduck_is_affiliate".into(), "false".into());

        let parsed = Facts::from_additional_info_map(&legacy, &score, "t")
            .expect("legacy map (no gizmoduck_labels) should still reconstruct");
        assert_eq!(parsed.user().labels, vec!["SpamHighRecall".to_owned()]);

        legacy.insert("gizmoduck_has_spam_high_recall".into(), "false".into());
        let parsed = Facts::from_additional_info_map(&legacy, &score, "t").unwrap();
        assert!(parsed.user().labels.is_empty());
    }

    #[test]
    fn from_additional_info_map_reconstructs_user_not_found_skip() {
        let score = ScoreResult {
            user_id: 99,
            ..Default::default()
        };
        let original = Facts {
            entity_type: EntityType::User,
            entity_id: 99,
            user_id: 99,
            topic: "t".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::User(UserFacts {
                allowlist: AllowlistFacts::default(),
                user: GizmoduckFacts::default(),
                cred: CredFacts {
                    is_high: Some(false),
                    ..Default::default()
                },
            }),
            uas: None,
        };
        let map = original.to_additional_info_map();
        let parsed = Facts::from_additional_info_map(&map, &score, "t")
            .expect("user_not_found skip-path map should reconstruct");

        assert!(!parsed.user_allowlist().is_allowlisted);
        assert!(!parsed.user().present);
        assert_eq!(parsed.cred().is_high, Some(false));
    }

    #[test]
    fn audit_map_round_trips_through_from_additional_info_map() {
        let score = ScoreResult {
            user_id: 12345,
            model_version: "v_test".into(),
            enforcement_note: "note".into(),
            ..Default::default()
        };
        let original = Facts {
            entity_type: EntityType::User,
            entity_id: 12345,
            user_id: 12345,
            topic: "t".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::User(UserFacts {
                allowlist: AllowlistFacts::default(),
                user: GizmoduckFacts {
                    present: true,
                    suspended: false,
                    deactivated: true,
                    labels: vec!["SpamHighRecall".into(), "RTBot".into()],
                    is_blue_verified: false,
                    verified_type: Some("Business".into()),
                    is_affiliate: false,
                    ..Default::default()
                },
                cred: CredFacts {
                    is_high: Some(false),
                    verified_type: None,
                    score: Some(32.5),
                    follower_count: Some(320),
                },
            }),
            uas: Some(r#"{"foo":"bar"}"#.into()),
        };
        let map = original.to_additional_info_map();
        assert!(!map.contains_key("allowlist_is_allowlisted"));
        let parsed = Facts::from_additional_info_map(&map, &score, "t")
            .expect("act-path map should reconstruct");

        assert_eq!(parsed.user_id, original.user_id);
        assert_eq!(parsed.user().suspended, original.user().suspended);
        assert_eq!(parsed.user().deactivated, original.user().deactivated);
        assert_eq!(parsed.user().labels, original.user().labels);
        assert_eq!(
            parsed.user().is_blue_verified,
            original.user().is_blue_verified
        );
        assert_eq!(parsed.user().verified_type, original.user().verified_type);
        assert_eq!(parsed.user().is_affiliate, original.user().is_affiliate);
        assert_eq!(parsed.cred().is_high, original.cred().is_high);
        assert_eq!(parsed.cred().verified_type, original.cred().verified_type);
        assert_eq!(parsed.cred().score, original.cred().score);
        assert_eq!(parsed.cred().follower_count, original.cred().follower_count);
        assert_eq!(parsed.uas, original.uas);
    }

    #[test]
    fn audit_map_format_matches_legacy_debug_strings() {
        let facts = Facts {
            entity_type: EntityType::User,
            entity_id: 1,
            user_id: 1,
            topic: "t".into(),
            score: ScoreFacts::from_score(&ScoreResult::default()),
            entity: EntityFacts::User(UserFacts {
                allowlist: AllowlistFacts::default(),
                user: GizmoduckFacts {
                    present: true,
                    verified_type: Some("Business".into()),
                    ..Default::default()
                },
                cred: CredFacts {
                    is_high: Some(false),
                    follower_count: Some(320),
                    score: Some(32.26),
                    ..Default::default()
                },
            }),
            uas: None,
        };
        let m = facts.to_additional_info_map();
        assert_eq!(m["gizmoduck_verified_type"], "Some(\"Business\")");
        assert_eq!(m["cred_is_high"], "Some(false)");
        assert_eq!(m["cred_follower_count"], "Some(320)");
        assert_eq!(m["cred_score"], "Some(32.26)");
    }

    #[test]
    fn resolve_target_defaults_legacy_user_id_to_user_entity() {
        let score = ScoreResult {
            user_id: 777,
            ..Default::default()
        };
        assert_eq!(Facts::resolve_target(&score), (EntityType::User, 777));
    }

    #[test]
    fn resolve_target_unknown_enum_value_falls_back_to_user() {
        let score = ScoreResult {
            user_id: 777,
            entity_type: 99,
            ..Default::default()
        };
        assert_eq!(Facts::resolve_target(&score), (EntityType::User, 777));
    }

    #[test]
    fn resolve_target_user_with_explicit_entity_id() {
        use xai_abuse_proto::enforcement::EntityType as ProtoEntityType;
        let score = ScoreResult {
            user_id: 777,
            entity_type: ProtoEntityType::User as i32,
            entity_id: 888,
            ..Default::default()
        };
        assert_eq!(Facts::resolve_target(&score), (EntityType::User, 888));
    }

    #[test]
    fn resolve_target_post_entity() {
        use xai_abuse_proto::enforcement::EntityType as ProtoEntityType;
        let score = ScoreResult {
            user_id: 777,
            entity_type: ProtoEntityType::Post as i32,
            entity_id: 555,
            ..Default::default()
        };
        assert_eq!(Facts::resolve_target(&score), (EntityType::Post, 555));
    }

    #[test]
    fn audit_map_emits_entity_keys_and_reconstructs() {
        use xai_abuse_proto::enforcement::EntityType as ProtoEntityType;
        let score = ScoreResult {
            user_id: 100,
            entity_type: ProtoEntityType::Post as i32,
            entity_id: 200,
            ..Default::default()
        };
        let (entity_type, entity_id) = Facts::resolve_target(&score);
        let facts = Facts {
            entity_type,
            entity_id,
            user_id: 100,
            topic: "t".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::Post(PostFacts {
                allowlist: AllowlistFacts {
                    is_allowlisted: true,
                    added_by: Some("oncall".into()),
                    reason: Some("false positive".into()),
                    ttl_secs: Some(3600),
                },
                present: true,
                author_id: 100,
                labels: vec!["spammy_post".into()],
                author: UserFacts {
                    allowlist: AllowlistFacts {
                        is_allowlisted: true,
                        added_by: Some("slackbot".into()),
                        reason: Some("vip author".into()),
                        ttl_secs: Some(7200),
                    },
                    cred: CredFacts {
                        is_high: Some(false),
                        follower_count: Some(42),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                invited: InvitedReply::NotAReply,
            }),
            uas: None,
        };
        let m = facts.to_additional_info_map();
        assert_eq!(m["entity_type"], "post");
        assert_eq!(m["entity_id"], "200");
        assert_eq!(m["post_present"], "true");
        assert_eq!(m["post_author_id"], "100");
        assert_eq!(m["post_labels"], r#"["spammy_post"]"#);
        assert_eq!(m["post_allowlist_is_allowlisted"], "true");
        assert_eq!(m["post_allowlist_added_by"], r#"Some("oncall")"#);
        assert_eq!(m["allowlist_is_allowlisted"], "true");
        assert_eq!(m["allowlist_added_by"], r#"Some("slackbot")"#);
        assert_eq!(m["cred_is_high"], "Some(false)");
        assert_eq!(m["cred_follower_count"], "Some(42)");

        let parsed = Facts::from_additional_info_map(&m, &score, "t").unwrap();
        assert_eq!(parsed.entity_type, EntityType::Post);
        assert_eq!(parsed.entity_id, 200);
        assert_eq!(parsed.cred().is_high, Some(false));
        assert_eq!(parsed.cred().follower_count, Some(42));
        assert!(parsed.user_allowlist().is_allowlisted);
        assert_eq!(
            parsed.user_allowlist().added_by.as_deref(),
            Some("slackbot")
        );
        assert_eq!(parsed.user_allowlist().ttl_secs, Some(7200));
        match parsed.entity {
            EntityFacts::Post(p) => {
                assert!(p.present);
                assert_eq!(p.author_id, 100);
                assert_eq!(p.labels, vec!["spammy_post".to_owned()]);
                assert!(p.allowlist.is_allowlisted);
                assert_eq!(p.allowlist.added_by.as_deref(), Some("oncall"));
                assert_eq!(p.allowlist.reason.as_deref(), Some("false positive"));
                assert_eq!(p.allowlist.ttl_secs, Some(3600));
            }
            other => panic!("expected Post entity, got {other:?}"),
        }
    }

    #[test]
    fn user_reconstruction_sets_user_id_to_entity_id_not_score_user_id() {
        let score = ScoreResult {
            user_id: 123,
            ..Default::default()
        };
        let original = Facts {
            entity_type: EntityType::User,
            entity_id: 888,
            user_id: 888,
            topic: "t".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::User(UserFacts::default()),
            uas: None,
        };
        let map = original.to_additional_info_map();
        let parsed = Facts::from_additional_info_map(&map, &score, "t").unwrap();
        assert_eq!(parsed.entity_id, 888);
        assert_eq!(
            parsed.user_id, 888,
            "user_id must follow entity_id, not score.user_id"
        );
    }

    #[test]
    fn from_additional_info_map_defaults_entity_to_user_for_legacy_entries() {
        let score = ScoreResult {
            user_id: 4242,
            ..Default::default()
        };
        let mut map = BTreeMap::new();
        map.insert("cred_is_high".into(), "Some(false)".into());
        let parsed = Facts::from_additional_info_map(&map, &score, "t").unwrap();
        assert_eq!(parsed.entity_type, EntityType::User);
        assert_eq!(parsed.entity_id, 4242);
    }

    #[test]
    fn uas_body_parses_prod_shape() {
        let raw = r#"{
          "userId": "1451154599834959874",
          "lastMonth": [
            {"clientAppId": "3033300", "activeSeconds": 392.06},
            {"clientAppId": "129032", "activeSeconds": 1768.37}
          ],
          "lastYear": [
            {"clientAppId": "3033300", "activeSeconds": 392.06},
            {"clientAppId": "129032", "activeSeconds": 1768.37}
          ]
        }"#;
        let body = UasBody::parse(raw).expect("prod shape should parse");
        assert_eq!(body.user_id.as_deref(), Some("1451154599834959874"));
        assert!(body.last_day.is_empty());
        assert!(body.last_week.is_empty());
        assert_eq!(body.last_month.len(), 2);
        assert_eq!(body.last_year.len(), 2);
        assert_eq!(body.last_month[0].client_app_id, "3033300");
        assert!((body.last_month[0].active_seconds - 392.06).abs() < 1e-9);
        assert_eq!(body.last_month[1].client_app_id, "129032");

        let rows: Vec<_> = body.activities().collect();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].0, "lastMonth");
        assert_eq!(rows[2].0, "lastYear");
    }

    #[test]
    fn uas_body_parse_rejects_garbage() {
        assert!(UasBody::parse("not json").is_none());
        assert!(UasBody::parse("").is_none());
    }

    #[test]
    fn uas_body_missing_periods_defaults_empty() {
        let body = UasBody::parse(r#"{"userId":"1"}"#).unwrap();
        assert!(body.last_day.is_empty());
        assert!(body.last_week.is_empty());
        assert!(body.last_month.is_empty());
        assert!(body.last_year.is_empty());
        assert_eq!(body.activities().count(), 0);
    }

    #[test]
    fn uas_body_parses_prod_shape_with_day_and_week() {
        let raw = r#"{
          "userId":"285734886",
          "lastDay":[{"clientAppId":"258901","activeSeconds":7029.860000000001}],
          "lastWeek":[
            {"clientAppId":"258901","activeSeconds":79952.78000000001},
            {"clientAppId":"3033300","activeSeconds":0.63}
          ],
          "lastMonth":[
            {"clientAppId":"258901","activeSeconds":393982.6199999999},
            {"clientAppId":"3033300","activeSeconds":14.26}
          ],
          "lastYear":[
            {"clientAppId":"258901","activeSeconds":4709903.989999999},
            {"clientAppId":"3033300","activeSeconds":121146.08000000007}
          ]
        }"#;
        let body = UasBody::parse(raw).expect("fou/prod four-period body must parse");
        assert_eq!(body.last_day.len(), 1);
        assert_eq!(body.last_week.len(), 2);
        assert_eq!(body.last_month.len(), 2);
        assert_eq!(body.last_year.len(), 2);
        let rows: Vec<_> = body.activities().collect();
        assert_eq!(rows.len(), 7, "all four periods are emitted");
        assert_eq!(rows[0].0, "lastDay");
        assert_eq!(rows[1].0, "lastWeek");
        assert_eq!(rows[3].0, "lastMonth");
        assert_eq!(rows[5].0, "lastYear");
        assert!((rows[0].1.active_seconds - 7029.86).abs() < 1e-6);
    }

    #[test]
    fn uas_body_parses_year_only_prod_shape() {
        let raw = r#"{"userId":"1716625813599068161","lastYear":[{"clientAppId":"3033300","activeSeconds":14.94}]}"#;
        let body = UasBody::parse(raw).expect("year-only must parse");
        assert!(body.last_day.is_empty());
        assert!(body.last_week.is_empty());
        assert!(body.last_month.is_empty());
        assert_eq!(body.last_year.len(), 1);
        assert_eq!(body.activities().count(), 1);
    }

    #[test]
    fn the_invited_answer_round_trips_and_only_replies_write_it() {
        let score = ScoreResult {
            user_id: 7,
            entity_type: xai_abuse_proto::enforcement::EntityType::Post as i32,
            entity_id: 70,
            ..Default::default()
        };
        for invited in [
            InvitedReply::Mentioned,
            InvitedReply::NotInvited,
            InvitedReply::Unknown,
            InvitedReply::NotAReply,
        ] {
            let facts = Facts {
                entity_type: EntityType::Post,
                entity_id: 70,
                user_id: 7,
                topic: "t".into(),
                score: ScoreFacts::from_score(&score),
                entity: EntityFacts::Post(PostFacts {
                    author_id: 7,
                    invited,
                    ..Default::default()
                }),
                uas: None,
            };
            let m = facts.to_additional_info_map();
            assert_eq!(
                m.get("post_invited_reply").map(String::as_str),
                (invited != InvitedReply::NotAReply).then_some(invited.as_str()),
            );
            let back = Facts::from_additional_info_map(&m, &score, "t").unwrap();
            let EntityFacts::Post(p) = back.entity else {
                panic!("post")
            };
            assert_eq!(p.invited, invited);
        }
    }
}
