use std::net::IpAddr;
use std::sync::LazyLock;

use cel::Program;
use serde::{Deserialize, Serialize};
use xai_abuse_enforcement_types::action_control::{
    self as types, ActionControlRecord, FollowerSkipView,
};
pub use xai_abuse_enforcement_types::action_control::{
    SafetyCheckView, SafetyChecksByEntity, SafetyChecksStatus,
};

use crate::facts::{
    EmailConfirmedState, EntityFacts, EntityType, Facts, GizmoduckFacts, InvitedReply,
};
use crate::rules::{CompiledRules, EvalError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyCheck {
    pub id: &'static str,
    pub reason: &'static str,
    pub when: &'static str,
}

pub const USER_CHECKS: &[SafetyCheck] = &[
    SafetyCheck {
        id: "user_in_allowlist",
        reason: "user_in_allowlist",
        when: "user_allowlist.is_allowlisted",
    },
    SafetyCheck {
        id: "user_not_found",
        reason: "user_not_found",
        when: "!user.present",
    },
    SafetyCheck {
        id: "high_follower_count",
        reason: "high_follower_count",
        when: "cred.follower_count >= 12.34",
    },
    SafetyCheck {
        id: "pagerank_skipped",
        reason: "pagerank_skipped",
        when: "cred.is_high || cred.score >= 12.34",
    },
];

pub const POST_CHECKS: &[SafetyCheck] = &[
    SafetyCheck {
        id: "post_in_allowlist",
        reason: "post_in_allowlist",
        when: "post_allowlist.is_allowlisted",
    },
    SafetyCheck {
        id: "user_in_allowlist",
        reason: "user_in_allowlist",
        when: "user_allowlist.is_allowlisted",
    },
    SafetyCheck {
        id: "user_not_found",
        reason: "user_not_found",
        when: "!user.present",
    },
    SafetyCheck {
        id: "high_follower_count",
        reason: "high_follower_count",
        when: "cred.follower_count >= 12.34 && !score.skip_author_credibility_prechecks",
    },
    SafetyCheck {
        id: "pagerank_skipped",
        reason: "pagerank_skipped",
        when: "(cred.is_high || cred.score >= 12.34) && !score.skip_author_credibility_prechecks",
    },
];

pub fn checks(entity: EntityType) -> &'static [SafetyCheck] {
    match entity {
        EntityType::User => USER_CHECKS,
        EntityType::Post => POST_CHECKS,
    }
}

fn compile(list: &'static [SafetyCheck]) -> Vec<Program> {
    list.iter()
        .map(|c| {
            Program::compile(c.when)
                .unwrap_or_else(|e| panic!("safety check {} does not compile: {e:?}", c.id))
        })
        .collect()
}

static USER_PROGRAMS: LazyLock<Vec<Program>> = LazyLock::new(|| compile(USER_CHECKS));
static POST_PROGRAMS: LazyLock<Vec<Program>> = LazyLock::new(|| compile(POST_CHECKS));

fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn status(entity: EntityType, rules: &CompiledRules) -> SafetyChecksStatus {
    let list = checks(entity);
    let leading: Vec<_> = rules
        .leading_skips()
        .into_iter()
        .filter(|r| !SERVICE_ONLY.contains(&r.id))
        .collect();
    let mut problems = Vec::new();
    let mut views = Vec::with_capacity(list.len());
    for (i, check) in list.iter().enumerate() {
        let status = match leading.iter().position(|r| r.id == check.id) {
            None => {
                problems.push(format!(
                    "the rules have no safety skip `{}` at the top (it was removed, renamed, or moved below a rule that acts)",
                    check.id
                ));
                "missing"
            }
            Some(at) => {
                let r = leading[at];
                if r.reason != check.reason || normalize(r.when) != normalize(check.when) {
                    problems.push(format!(
                        "safety skip `{}` changed: the rules say `{}` → `{}`, the service expects `{}` → `{}`",
                        check.id,
                        normalize(r.when),
                        r.reason,
                        check.when,
                        check.reason
                    ));
                    "changed"
                } else if at != i {
                    problems.push(format!(
                        "safety skip `{}` moved: position {} in the rules, {} expected",
                        check.id,
                        at + 1,
                        i + 1
                    ));
                    "moved"
                } else {
                    "ok"
                }
            }
        };
        views.push(SafetyCheckView {
            id: check.id.into(),
            reason: check.reason.into(),
            when: check.when.into(),
            status: status.into(),
        });
    }
    for r in &leading {
        if !list.iter().any(|c| c.id == r.id) {
            problems.push(format!(
                "the rules start with a safety skip `{}` the service doesn't know; add it to the service's list",
                r.id
            ));
        }
    }
    SafetyChecksStatus {
        ok: problems.is_empty(),
        problems,
        checks: views,
    }
}

pub const FOLLOWER_CEILING: SafetyCheck = SafetyCheck {
    id: types::FOLLOWER_CEILING_CHECK.id,
    reason: "very_high_follower_count",
    when: "the account's follower count is at or above the follower ceiling, or the head's raised limit",
};

static FOLLOWER_CEILING_COUNT: LazyLock<f64> =
    LazyLock::new(|| f64::from(types::follower_ceiling()));

pub(crate) const PRE_CEILING: &[&str] =
    &["post_in_allowlist", "user_in_allowlist", "user_not_found"];

pub const GIZMODUCK_TEST_USER: SafetyCheck = SafetyCheck {
    id: "gizmoduck_test_user",
    reason: "test_user_skipped",
    when: "gizmoduck marks the account as a temporary or permanent test account",
};

pub const TEST_ACCOUNT: SafetyCheck = SafetyCheck {
    id: "test_account",
    reason: "test_user_skipped",
    when: "the account matches the service's test-account policy",
};

pub const INVITED_REPLY: SafetyCheck = SafetyCheck {
    id: "invited_reply",
    reason: "invited_reply_skipped",
    when: "the reply's parent post mentions the reply's author in its own text",
};

pub const INVITED_REPLY_UNKNOWN: SafetyCheck = SafetyCheck {
    id: "invited_reply_unknown",
    reason: "invited_reply_unknown_skipped",
    when: "the reply's parent post couldn't be read",
};

const SERVICE_ONLY: &[&str] = &[
    FOLLOWER_CEILING.id,
    GIZMODUCK_TEST_USER.id,
    TEST_ACCOUNT.id,
    INVITED_REPLY.id,
    INVITED_REPLY_UNKNOWN.id,
];

pub fn invited_reply_stop(facts: &Facts) -> Option<&'static SafetyCheck> {
    let EntityFacts::Post(post) = &facts.entity else {
        return None;
    };
    match post.invited {
        InvitedReply::Mentioned => Some(&INVITED_REPLY),
        InvitedReply::Unknown => Some(&INVITED_REPLY_UNKNOWN),
        InvitedReply::NotAReply | InvitedReply::NotInvited => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestAccountSignal {
    TestUserStatus,
    Role,
    EmailDomain,
    CreationCidr,
}

impl TestAccountSignal {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TestUserStatus => "test_user_status",
            Self::Role => "test_role",
            Self::EmailDomain => "test_email_domain",
            Self::CreationCidr => "test_creation_cidr",
        }
    }

    fn check(self) -> &'static SafetyCheck {
        match self {
            Self::TestUserStatus => &GIZMODUCK_TEST_USER,
            Self::Role | Self::EmailDomain | Self::CreationCidr => &TEST_ACCOUNT,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TestAccountPolicy {
    email_domains: Vec<String>,
    roles: Vec<String>,
    creation_cidrs: Vec<IpNet>,
}

impl TestAccountPolicy {
                pub fn new(
        email_domains: &[String],
        roles: &[String],
        creation_cidrs: &[String],
    ) -> Result<Self, String> {
        let entries = |list: &[String]| -> Vec<String> {
            list.iter()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        };
        let email_domains = entries(email_domains)
            .into_iter()
            .map(|d| {
                let valid = d.is_ascii()
                    && !d.starts_with('.')
                    && !d.ends_with('.')
                    && d.contains('.')
                    && d.chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-');
                if valid {
                    Ok(d.to_ascii_lowercase())
                } else {
                    Err(format!(
                        "test-account e-mail domain `{d}` is not a domain name"
                    ))
                }
            })
            .collect::<Result<_, _>>()?;
        let creation_cidrs = entries(creation_cidrs)
            .iter()
            .map(|c| IpNet::parse(c))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            email_domains,
            roles: entries(roles),
            creation_cidrs,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.email_domains.is_empty() && self.roles.is_empty() && self.creation_cidrs.is_empty()
    }

            pub fn sizes(&self) -> (usize, usize, usize) {
        (
            self.email_domains.len(),
            self.roles.len(),
            self.creation_cidrs.len(),
        )
    }

            pub fn signal(&self, user: &GizmoduckFacts) -> Option<TestAccountSignal> {
        if is_test_status(user) {
            return Some(TestAccountSignal::TestUserStatus);
        }
        if user.roles.iter().any(|r| self.roles.contains(r)) {
            return Some(TestAccountSignal::Role);
        }
        if user.email_confirmed == Some(EmailConfirmedState::Confirmed)
            && let Some(domain) = user.login_email.as_deref().and_then(email_domain)
            && self.email_domains.contains(&domain)
        {
            return Some(TestAccountSignal::EmailDomain);
        }
        if let Some(ip) = user.creation_ip.as_deref().and_then(parse_ip)
            && self.creation_cidrs.iter().any(|net| net.contains(ip))
        {
            return Some(TestAccountSignal::CreationCidr);
        }
        None
    }
}

fn is_test_status(user: &GizmoduckFacts) -> bool {
    matches!(
        user.test_user_status.as_deref(),
        Some("Temporary" | "Permanent")
    )
}

fn email_domain(email: &str) -> Option<String> {
    let (local, domain) = email.trim().split_once('@')?;
    if local.is_empty() || domain.is_empty() || domain.contains('@') {
        return None;
    }
    Some(domain.to_ascii_lowercase())
}

fn parse_ip(s: &str) -> Option<IpAddr> {
    let s = s.trim();
    let s = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(s);
    s.parse::<IpAddr>().ok().map(|ip| ip.to_canonical())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IpNet {
    addr: IpAddr,
    prefix: u8,
}

impl IpNet {
    fn parse(s: &str) -> Result<Self, String> {
        let bad = || format!("test-account creation network `{s}` is not an IP network");
        let (addr, prefix) = match s.split_once('/') {
            Some((a, p)) => (a, Some(p.parse::<u8>().map_err(|_| bad())?)),
            None => (s, None),
        };
        let addr: IpAddr = addr.parse().map_err(|_| bad())?;
        let max = if addr.is_ipv4() { 32 } else { 128 };
        let prefix = prefix.unwrap_or(max);
        if prefix > max {
            return Err(bad());
        }
        let (addr, prefix) = match addr {
            IpAddr::V6(v6) if prefix >= 96 => match v6.to_ipv4_mapped() {
                Some(v4) => (IpAddr::V4(v4), prefix - 96),
                None => (addr, prefix),
            },
            _ => (addr, prefix),
        };
        Ok(Self { addr, prefix })
    }

    fn contains(&self, ip: IpAddr) -> bool {
        match (self.addr, ip.to_canonical()) {
            (IpAddr::V4(net), IpAddr::V4(ip)) => {
                let mask = u32::MAX
                    .checked_shl(32 - u32::from(self.prefix))
                    .unwrap_or(0);
                u32::from(net) & mask == u32::from(ip) & mask
            }
            (IpAddr::V6(net), IpAddr::V6(ip)) => {
                let mask = u128::MAX
                    .checked_shl(128 - u32::from(self.prefix))
                    .unwrap_or(0);
                u128::from(net) & mask == u128::from(ip) & mask
            }
            _ => false,
        }
    }
}

pub fn test_account_stop(facts: &Facts) -> Option<&'static SafetyCheck> {
    test_account_check(facts.user())
}

pub fn test_account_check(user: &GizmoduckFacts) -> Option<&'static SafetyCheck> {
    if is_test_status(user) {
        return Some(&GIZMODUCK_TEST_USER);
    }
    user.test_signal.map(TestAccountSignal::check)
}

pub fn test_account_signal_kind(facts: &Facts) -> Option<&'static str> {
    let user = facts.user();
    if is_test_status(user) {
        return Some(TestAccountSignal::TestUserStatus.as_str());
    }
    user.test_signal.map(TestAccountSignal::as_str)
}

pub fn ceiling_stops(facts: &Facts, page_driven: bool) -> bool {
    stops_at(facts, page_driven, *FOLLOWER_CEILING_COUNT)
}

fn stops_at(facts: &Facts, page_driven: bool, ceiling: f64) -> bool {
    match facts.cred().follower_count {
        Some(n) => n as f64 >= ceiling,
        None => page_driven && facts.entity_type == EntityType::Post,
    }
}

const FOLLOWER_SKIP: &str = "high_follower_count";
const CREDIBILITY_SKIP: &str = "pagerank_skipped";

fn followers_in(when: &str) -> f64 {
    when.split(">=")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("`{when}` is not `cred.follower_count >= N ...`"))
}

static DEFAULT_FOLLOWER_SKIP: LazyLock<f64> = LazyLock::new(|| {
    let user = USER_CHECKS.iter().find(|c| c.id == FOLLOWER_SKIP);
    let post = POST_CHECKS.iter().find(|c| c.id == FOLLOWER_SKIP);
    let n = followers_in(user.expect("the user list has a follower skip").when);
    assert_eq!(
        n,
        followers_in(post.expect("the post list has a follower skip").when),
        "the user and post follower skips start at the same count"
    );
    n
});

pub fn follower_skip_view() -> FollowerSkipView {
    FollowerSkipView {
        today_threshold: DEFAULT_FOLLOWER_SKIP.ceil() as u64,
        ceiling: follower_ceiling(),
        max_threshold: u64::from(types::FOLLOWER_THRESHOLD_MAX),
    }
}

pub fn follower_ceiling() -> u64 {
    u64::from(types::follower_ceiling())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AuthorSkips {
    pub follower_threshold: Option<u32>,
    pub skip_high_credibility: Option<bool>,
}

impl AuthorSkips {
            pub fn of(record: &ActionControlRecord) -> Self {
        Self {
            follower_threshold: record.follower_threshold.map(|n| match n {
                0 => types::stored_off_follower_threshold(),
                n => n,
            }),
            skip_high_credibility: record.skip_high_credibility,
        }
    }

        pub fn follower_floor(self) -> f64 {
        match self.follower_threshold {
            None => DEFAULT_FOLLOWER_SKIP.ceil(),
            Some(0) => f64::from(types::stored_off_follower_threshold()),
            Some(n) => f64::from(n),
        }
    }

            pub fn follower_ceiling(self) -> f64 {
        self.follower_floor().max(*FOLLOWER_CEILING_COUNT)
    }

    pub fn skips_high_credibility(self) -> bool {
        self.skip_high_credibility != Some(false)
    }

            pub fn harsher_than(self, before: Self) -> Vec<String> {
        let mut why = Vec::new();
        let floor = |s: Self| s.follower_floor();
        if floor(self) > floor(before) {
            let show = |s: Self| format!("{}+", s.follower_floor().ceil() as u64);
            why.push(format!(
                "skips fewer high-follower accounts ({} to {})",
                show(before),
                show(self)
            ));
        }
        if self.follower_ceiling() > before.follower_ceiling() {
            why.push(format!(
                "raises the follower ceiling ({}+ to {}+), acting on accounts every other head skips",
                before.follower_ceiling() as u64,
                self.follower_ceiling() as u64
            ));
        }
        if before.skips_high_credibility() && !self.skips_high_credibility() {
            why.push("stops skipping high-credibility accounts".to_owned());
        }
        why
    }
}

pub fn first_page_stop(
    entity: EntityType,
    facts: &Facts,
    skips: AuthorSkips,
) -> Result<Option<&'static SafetyCheck>, EvalError> {
    let listed = if skips == AuthorSkips::default() {
        first_stop(entity, facts)
    } else {
        first_stop_with(entity, facts, skips)
    };
    if let Ok(Some(check)) = &listed
        && PRE_CEILING.contains(&check.id)
    {
        return listed;
    }
    if let Some(check) = test_account_stop(facts) {
        return Ok(Some(check));
    }
    if stops_at(facts, true, skips.follower_ceiling()) {
        return Ok(Some(&FOLLOWER_CEILING));
    }
    match listed {
        Ok(None) => Ok(invited_reply_stop(facts)),
        listed => listed,
    }
}

pub fn first_stop(
    entity: EntityType,
    facts: &Facts,
) -> Result<Option<&'static SafetyCheck>, EvalError> {
    let (list, programs) = match entity {
        EntityType::User => (USER_CHECKS, &*USER_PROGRAMS),
        EntityType::Post => (POST_CHECKS, &*POST_PROGRAMS),
    };
    let pairs: Vec<(&str, &Program)> = list.iter().map(|c| c.id).zip(programs.iter()).collect();
    Ok(crate::rules::first_true(&pairs, facts)?.map(|i| &list[i]))
}

fn first_stop_with(
    entity: EntityType,
    facts: &Facts,
    skips: AuthorSkips,
) -> Result<Option<&'static SafetyCheck>, EvalError> {
    let (list, programs) = match entity {
        EntityType::User => (USER_CHECKS, &*USER_PROGRAMS),
        EntityType::Post => (POST_CHECKS, &*POST_PROGRAMS),
    };
    let listed =
        |range: std::ops::Range<usize>| -> Result<Option<&'static SafetyCheck>, EvalError> {
            let kept: Vec<usize> = range
                .filter(|&i| list[i].id != CREDIBILITY_SKIP || skips.skips_high_credibility())
                .collect();
            if kept.is_empty() {
                return Ok(None);
            }
            let pairs: Vec<(&str, &Program)> =
                kept.iter().map(|&i| (list[i].id, &programs[i])).collect();
            Ok(crate::rules::first_true(&pairs, facts)?.map(|k| &list[kept[k]]))
        };
    let Some(floor_at) = list.iter().position(|c| c.id == FOLLOWER_SKIP) else {
        return listed(0..list.len());
    };
    if let Some(stop) = listed(0..floor_at)? {
        return Ok(Some(stop));
    }
    let waived = entity == EntityType::Post && facts.score.skip_author_credibility_prechecks;
    let followers = facts.cred().follower_count.unwrap_or(0) as f64;
    if !waived && followers >= skips.follower_floor() {
        return Ok(Some(&list[floor_at]));
    }
    listed(floor_at + 1..list.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAKED_USER: &str = include_str!("../rules/enforcement_user.yaml");
    const BAKED_POST: &str = include_str!("../rules/enforcement_post.yaml");

    fn compiled(yaml: &str) -> CompiledRules {
        CompiledRules::from_yaml(yaml).unwrap()
    }

    #[test]
    fn baked_rules_match_the_list() {
        let user = status(EntityType::User, &compiled(BAKED_USER));
        assert!(user.ok, "{:?}", user.problems);
        assert!(user.checks.iter().all(|c| c.status == "ok"));
        let post = status(EntityType::Post, &compiled(BAKED_POST));
        assert!(post.ok, "{:?}", post.problems);
        assert_eq!(post.checks.len(), 5);
    }

    #[test]
    fn every_check_compiles() {
        assert_eq!(USER_PROGRAMS.len(), USER_CHECKS.len());
        assert_eq!(POST_PROGRAMS.len(), POST_CHECKS.len());
        assert!(FOLLOWER_CEILING_COUNT.is_finite());
    }

        fn edited_user(edit: impl Fn(&str) -> String) -> SafetyChecksStatus {
        let yaml = edit(BAKED_USER);
        status(EntityType::User, &compiled(&yaml))
    }

    fn rule_block(yaml: &str, id: &str) -> (usize, usize) {
        let start = yaml.find(&format!("  - id: {id}\n")).unwrap();
        let end = yaml[start + 3..]
            .find("\n  - id: ")
            .map_or(yaml.len(), |e| start + 3 + e + 1);
        (start, end)
    }

    fn without(yaml: &str, id: &str) -> String {
        let (s, e) = rule_block(yaml, id);
        format!("{}{}", &yaml[..s], &yaml[e..])
    }

        fn in_block(yaml: &str, id: &str, from: &str, to: &str) -> String {
        let (s, e) = rule_block(yaml, id);
        let block = yaml[s..e].replacen(from, to, 1);
        assert_ne!(block, yaml[s..e], "`{from}` is not in rule {id}");
        format!("{}{}{}", &yaml[..s], block, &yaml[e..])
    }

    #[test]
    fn a_missing_check_is_uncertain() {
        let st = edited_user(|y| without(y, "high_follower_count"));
        assert!(!st.ok);
        assert_eq!(
            st.checks
                .iter()
                .find(|c| c.id == "high_follower_count")
                .unwrap()
                .status,
            "missing"
        );
    }

    #[test]
    fn a_renamed_check_is_uncertain() {
        let st = edited_user(|y| {
            y.replacen(
                "  - id: high_follower_count\n",
                "  - id: high_followers\n",
                1,
            )
        });
        assert!(!st.ok);
        assert!(
            st.problems
                .iter()
                .any(|p| p.contains("high_follower_count"))
        );
        assert!(st.problems.iter().any(|p| p.contains("high_followers")));
    }

    #[test]
    fn a_reordered_check_is_uncertain() {
        let st = edited_user(|y| {
            let (s, e) = rule_block(y, "high_follower_count");
            let block = y[s..e].to_owned();
            let rest = format!("{}{}", &y[..s], &y[e..]);
            let (_, after) = rule_block(&rest, "pagerank_skipped");
            format!("{}{}{}", &rest[..after], block, &rest[after..])
        });
        assert!(!st.ok);
        assert!(st.checks.iter().any(|c| c.status == "moved"), "{st:?}");
    }

    #[test]
    fn a_check_moved_below_an_act_rule_is_missing() {
        let st = edited_user(|y| {
            let (s, e) = rule_block(y, "pagerank_skipped");
            let block = y[s..e].to_owned();
            let rest = format!("{}{}", &y[..s], &y[e..]);
            let (_, after) = rule_block(&rest, "anchor_campaign_suspend");
            format!("{}{}{}", &rest[..after], block, &rest[after..])
        });
        assert!(!st.ok);
        assert_eq!(
            st.checks
                .iter()
                .find(|c| c.id == "pagerank_skipped")
                .unwrap()
                .status,
            "missing"
        );
    }

    #[test]
    fn a_changed_condition_or_reason_is_uncertain() {
        let st = edited_user(|y| in_block(y, "high_follower_count", " >= ", " > "));
        assert!(!st.ok);
        assert_eq!(
            st.checks
                .iter()
                .find(|c| c.id == "high_follower_count")
                .unwrap()
                .status,
            "changed"
        );
        let st = edited_user(|y| {
            y.replacen(
                "      reason: user_not_found\n",
                "      reason: missing_user\n",
                1,
            )
        });
        assert!(!st.ok);
    }

    #[test]
    fn whitespace_in_a_condition_is_not_a_change() {
        let st = edited_user(|y| in_block(y, "high_follower_count", " >= ", "   >=   "));
        assert!(st.ok, "{:?}", st.problems);
    }

    #[test]
    fn a_new_leading_skip_is_uncertain() {
        let st = edited_user(|y| {
            y.replacen(
                "rules:\n",
                "rules:\n  - id: new_global_skip\n    when: user.deactivated\n    then: { kind: skip, reason: deactivated }\n",
                1,
            )
        });
        assert!(!st.ok);
        assert!(st.problems.iter().any(|p| p.contains("new_global_skip")));
    }

    #[test]
    fn malformed_rules_never_reach_the_comparison() {
        assert!(CompiledRules::from_yaml("for_entity: user\nrules: [{id: x}]").is_err());
    }

    #[test]
    fn checks_stop_in_list_order_with_the_rules_reason() {
        use crate::facts::{CredFacts, EntityFacts, GizmoduckFacts, ScoreFacts, UserFacts};
        let facts = |present: bool, followers: i64| Facts {
            entity_type: EntityType::User,
            entity_id: 1,
            user_id: 1,
            topic: "t".into(),
            score: ScoreFacts::default(),
            entity: EntityFacts::User(UserFacts {
                user: GizmoduckFacts {
                    present,
                    ..Default::default()
                },
                cred: CredFacts {
                    follower_count: Some(followers),
                    ..Default::default()
                },
                ..Default::default()
            }),
            uas: None,
        };
        let floor = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let above = FOLLOWER_CEILING_COUNT.ceil() as i64 + 1;
        let stop = |f: &Facts| first_stop(EntityType::User, f).unwrap().map(|c| c.reason);
        assert_eq!(stop(&facts(true, floor - 1)), None);
        assert_eq!(stop(&facts(true, floor)), Some("high_follower_count"));
        assert_eq!(
            stop(&facts(true, above)),
            Some("high_follower_count"),
            "the ceiling is not in the list"
        );
        assert_eq!(stop(&facts(false, above)), Some("user_not_found"));
        let page = |f: &Facts| {
            first_page_stop(EntityType::User, f, AuthorSkips::default())
                .unwrap()
                .map(|c| c.reason)
        };
        assert_eq!(page(&facts(true, above)), Some("very_high_follower_count"));
        assert_eq!(page(&facts(false, above)), Some("user_not_found"));
    }

                #[test]
    fn a_ceiling_rule_in_the_rules_is_not_compared() {
        let rule = "  - id: very_high_follower_count\n    when: cred.follower_count >= 12.34\n    then:\n      kind: skip\n      reason: very_high_follower_count\n\n";
        for before in ["user_not_found", "high_follower_count", "user_in_allowlist"] {
            let st = edited_user(|y| {
                let (s, _) = rule_block(y, before);
                format!("{}{}{}", &y[..s], rule, &y[s..])
            });
            assert!(st.ok, "before {before}: {:?}", st.problems);
        }
        let post = status(
            EntityType::Post,
            &compiled(&{
                let (s, _) = rule_block(BAKED_POST, "high_follower_count");
                format!("{}{}{}", &BAKED_POST[..s], rule, &BAKED_POST[s..])
            }),
        );
        assert!(post.ok, "{:?}", post.problems);
        assert!(!BAKED_USER.contains("id: very_high_follower_count"));
        assert!(!BAKED_POST.contains("id: very_high_follower_count"));
    }

    #[test]
    fn page_post_decisions_stop_at_the_author_ceiling_whatever_the_waiver() {
        use crate::facts::{
            CredFacts, EntityFacts, GizmoduckFacts, PostFacts, ScoreFacts, UserFacts,
        };
        let facts = |followers: Option<i64>| Facts {
            entity_type: EntityType::Post,
            entity_id: 1,
            user_id: 2,
            topic: "t".into(),
            score: ScoreFacts {
                skip_author_credibility_prechecks: true,
                ..Default::default()
            },
            entity: EntityFacts::Post(PostFacts {
                author: UserFacts {
                    user: GizmoduckFacts {
                        present: true,
                        ..Default::default()
                    },
                    cred: CredFacts {
                        follower_count: followers,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            }),
            uas: None,
        };
        let page = |f: &Facts| {
            first_page_stop(EntityType::Post, f, AuthorSkips::default())
                .unwrap()
                .map(|c| c.reason)
        };
        let ceiling = FOLLOWER_CEILING_COUNT.ceil() as i64;
        assert_eq!(
            page(&facts(Some(ceiling - 1))),
            None,
            "waived below the ceiling"
        );
        assert_eq!(
            page(&facts(Some(ceiling))),
            Some("very_high_follower_count")
        );
        assert_eq!(
            page(&facts(None)),
            Some("very_high_follower_count"),
            "unknown count"
        );
        assert_eq!(
            first_stop(EntityType::Post, &facts(Some(ceiling + 1))).unwrap(),
            None,
            "the rules' own list is unchanged"
        );
    }

        #[derive(Clone, Copy, Default)]
    struct Author {
        followers: Option<i64>,
        high_cred: bool,
        waived: bool,
        allowlisted: bool,
    }

    fn author_facts(entity: EntityType, a: Author, labels: &[&str]) -> Facts {
        use crate::facts::{
            AllowlistFacts, CredFacts, EntityFacts, GizmoduckFacts, PostFacts, ScoreFacts,
            UserFacts,
        };
        let account = UserFacts {
            allowlist: AllowlistFacts {
                is_allowlisted: a.allowlisted,
                ..Default::default()
            },
            user: GizmoduckFacts {
                present: true,
                ..Default::default()
            },
            cred: CredFacts {
                follower_count: a.followers,
                is_high: Some(a.high_cred),
                ..Default::default()
            },
        };
        Facts {
            entity_type: entity,
            entity_id: 1,
            user_id: 2,
            topic: "t".into(),
            score: ScoreFacts {
                skip_author_credibility_prechecks: a.waived,
                labels: labels.iter().map(|l| (*l).to_owned()).collect(),
                ..Default::default()
            },
            entity: match entity {
                EntityType::User => EntityFacts::User(account),
                EntityType::Post => EntityFacts::Post(PostFacts {
                    author: account,
                    ..Default::default()
                }),
            },
            uas: None,
        }
    }

    fn skips(follower_threshold: Option<u32>, skip_high_credibility: Option<bool>) -> AuthorSkips {
        AuthorSkips {
            follower_threshold,
            skip_high_credibility,
        }
    }

    fn page_stop(entity: EntityType, a: Author, s: AuthorSkips) -> Option<&'static str> {
        first_page_stop(entity, &author_facts(entity, a, &[]), s)
            .unwrap()
            .map(|c| c.reason)
    }

    fn with_followers(n: i64) -> Author {
        Author {
            followers: Some(n),
            ..Default::default()
        }
    }

    const BOTH: [EntityType; 2] = [EntityType::User, EntityType::Post];

    #[test]
    fn the_default_and_the_ceiling_come_from_the_lists() {
        let view = follower_skip_view();
        assert_eq!(view.today_threshold, DEFAULT_FOLLOWER_SKIP.ceil() as u64);
        assert_eq!(view.ceiling, follower_ceiling());
        assert!(view.today_threshold < view.ceiling);
    }

    #[test]
    fn absent_skips_decide_exactly_as_the_list() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let c = follower_ceiling() as i64;
        let mut authors: Vec<Author> = [None, Some(0), Some(d - 1), Some(d), Some(c - 1), Some(c)]
            .into_iter()
            .map(|followers| Author {
                followers,
                ..Default::default()
            })
            .collect();
        let more: Vec<Author> = authors
            .iter()
            .flat_map(|a| {
                [
                    Author {
                        high_cred: true,
                        ..*a
                    },
                    Author { waived: true, ..*a },
                    Author {
                        allowlisted: true,
                        ..*a
                    },
                ]
            })
            .collect();
        authors.extend(more);
        let today = [
            skips(None, Some(true)),
            skips(Some(d as u32), None),
            skips(Some(d as u32), Some(true)),
        ];
        for entity in BOTH {
            for a in &authors {
                let f = author_facts(entity, *a, &[]);
                let listed = first_stop(entity, &f).unwrap().map(|c| c.reason);
                for s in today {
                    let rebuilt = first_stop_with(entity, &f, s).unwrap().map(|c| c.reason);
                    assert_eq!(
                        listed, rebuilt,
                        "{entity:?} {s:?} followers={:?} cred={} waived={} allowlisted={}",
                        a.followers, a.high_cred, a.waived, a.allowlisted
                    );
                }
            }
        }
    }

    #[test]
    fn a_stored_zero_threshold_reads_as_one_below_the_ceiling() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let c = follower_ceiling() as i64;
        let stored: ActionControlRecord = serde_json::from_value(serde_json::json!({
            "v": 5, "entity_type": "post", "head": "h", "mode": "live", "version": 3,
            "follower_threshold": 0,
        }))
        .unwrap();
        assert_eq!(
            AuthorSkips::of(&stored).follower_threshold,
            Some(types::stored_off_follower_threshold())
        );
        assert_eq!(i64::from(types::stored_off_follower_threshold()), c - 1);
        for zero in [skips(Some(0), None), AuthorSkips::of(&stored)] {
            assert_eq!(zero.follower_floor(), (c - 1) as f64);
            assert_eq!(zero.follower_ceiling(), c as f64);
            for entity in BOTH {
                assert_eq!(
                    page_stop(entity, with_followers(d), zero),
                    None,
                    "{entity:?}"
                );
                assert_eq!(
                    page_stop(entity, with_followers(c - 2), zero),
                    None,
                    "{entity:?}"
                );
                assert_eq!(
                    page_stop(entity, with_followers(c - 1), zero),
                    Some("high_follower_count"),
                    "{entity:?}: the one count whose decision changes"
                );
                assert_eq!(
                    page_stop(entity, with_followers(c), zero),
                    Some("very_high_follower_count"),
                    "{entity:?}"
                );
                let cred = Author {
                    high_cred: true,
                    ..with_followers(d)
                };
                assert_eq!(
                    page_stop(entity, cred, zero),
                    Some("pagerank_skipped"),
                    "{entity:?}"
                );
            }
        }
    }

    #[test]
    fn a_raised_limit_moves_the_ceiling_for_that_head_only() {
        let c = follower_ceiling() as i64;
        let raised = 250_000_i64;
        let s = skips(Some(raised as u32), None);
        assert_eq!(s.follower_ceiling(), raised as f64);
        for entity in BOTH {
            for n in [c, c + 1, raised - 1] {
                assert_eq!(
                    page_stop(entity, with_followers(n), s),
                    None,
                    "{entity:?} {n}"
                );
            }
            assert_eq!(
                page_stop(entity, with_followers(raised), s),
                Some("very_high_follower_count"),
                "{entity:?}: the head's number is the ceiling, same reason"
            );
            assert_eq!(
                page_stop(entity, with_followers(c), AuthorSkips::default()),
                Some("very_high_follower_count"),
                "{entity:?}: other heads keep the ceiling"
            );
        }
        let waived = |n: i64| Author {
            waived: true,
            ..with_followers(n)
        };
        assert_eq!(page_stop(EntityType::Post, waived(c), s), None);
        assert_eq!(
            page_stop(EntityType::Post, waived(raised), s),
            Some("very_high_follower_count"),
            "the producer waiver can't lift the raised limit"
        );
        assert_eq!(
            page_stop(EntityType::Post, Author::default(), s),
            Some("very_high_follower_count"),
            "an unknown count still stops a post"
        );
        let top = skips(Some(types::FOLLOWER_THRESHOLD_MAX), None);
        assert_eq!(
            page_stop(
                EntityType::User,
                with_followers(i64::from(types::FOLLOWER_THRESHOLD_MAX)),
                top
            ),
            Some("very_high_follower_count")
        );
        let f = author_facts(EntityType::User, with_followers(c), &[]);
        assert!(
            ceiling_stops(&f, false),
            "the rules always stop at the ceiling"
        );
    }

    #[test]
    fn a_threshold_skips_from_that_count_up() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let c = follower_ceiling() as i64;
        for entity in BOTH {
            for n in [1_i64, 250, d - 1, d, d + 1, c - 1] {
                let s = skips(Some(n as u32), None);
                assert_eq!(
                    page_stop(entity, with_followers(n - 1), s),
                    None,
                    "{entity:?} {n}"
                );
                assert_eq!(
                    page_stop(entity, with_followers(n), s),
                    Some("high_follower_count"),
                    "{entity:?} {n}"
                );
            }
            assert_eq!(
                page_stop(EntityType::User, Author::default(), skips(Some(1), None)),
                None,
                "an unknown count reads as 0 on an account"
            );
            assert_eq!(
                page_stop(EntityType::Post, Author::default(), skips(Some(0), None)),
                Some("very_high_follower_count"),
                "and stops a post at the ceiling"
            );
        }
        assert_eq!(
            page_stop(EntityType::User, with_followers(c), skips(Some(1), None)),
            Some("very_high_follower_count")
        );
        assert_eq!(
            page_stop(EntityType::Post, with_followers(c), skips(Some(1), None)),
            Some("very_high_follower_count")
        );
    }

    #[test]
    fn the_producer_waiver_lifts_both_skips_on_posts_only() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let c = follower_ceiling() as i64;
        let waived = |n: i64| Author {
            waived: true,
            ..with_followers(n)
        };
        let waived_cred = Author {
            high_cred: true,
            ..waived(1)
        };
        for s in [
            AuthorSkips::default(),
            skips(Some(250), None),
            skips(Some(0), Some(true)),
            skips(None, Some(true)),
        ] {
            assert_eq!(page_stop(EntityType::Post, waived(d), s), None, "{s:?}");
            assert_eq!(page_stop(EntityType::Post, waived(250), s), None, "{s:?}");
            assert_eq!(page_stop(EntityType::Post, waived_cred, s), None, "{s:?}");
            assert_eq!(
                page_stop(EntityType::Post, waived(c), s),
                Some("very_high_follower_count"),
                "{s:?}"
            );
        }
        assert_eq!(
            page_stop(EntityType::User, waived(d), AuthorSkips::default()),
            Some("high_follower_count"),
            "an account decision has no waiver"
        );
        assert_eq!(
            page_stop(EntityType::User, waived(250), skips(Some(250), None)),
            Some("high_follower_count")
        );
        assert_eq!(
            page_stop(EntityType::User, waived_cred, skips(Some(0), Some(true))),
            Some("pagerank_skipped")
        );
    }

    #[test]
    fn the_credibility_skip_follows_the_head() {
        let cred = Author {
            high_cred: true,
            ..with_followers(1)
        };
        for entity in BOTH {
            assert_eq!(
                page_stop(entity, cred, AuthorSkips::default()),
                Some("pagerank_skipped")
            );
            assert_eq!(
                page_stop(entity, cred, skips(None, Some(true))),
                Some("pagerank_skipped")
            );
            assert_eq!(
                page_stop(entity, cred, skips(None, Some(false))),
                None,
                "{entity:?}"
            );
            assert_eq!(
                page_stop(entity, cred, skips(Some(0), Some(false))),
                None,
                "{entity:?}"
            );
            let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
            let both = Author {
                high_cred: true,
                ..with_followers(d)
            };
            assert_eq!(
                page_stop(entity, both, skips(None, Some(false))),
                Some("high_follower_count"),
                "{entity:?}"
            );
        }
    }

    #[test]
    fn the_checks_before_the_follower_skip_still_stop_first() {
        let off = skips(Some(0), Some(false));
        for entity in BOTH {
            let allowlisted = Author {
                allowlisted: true,
                ..with_followers(1)
            };
            assert_eq!(
                page_stop(entity, allowlisted, off),
                Some("user_in_allowlist")
            );
            let mut f = author_facts(entity, with_followers(1), &[]);
            match &mut f.entity {
                crate::facts::EntityFacts::User(u) => u.user.present = false,
                crate::facts::EntityFacts::Post(p) => p.author.user.present = false,
            }
            assert_eq!(
                first_page_stop(entity, &f, off).unwrap().map(|c| c.reason),
                Some("user_not_found")
            );
        }
    }

    #[test]
    fn no_routing_label_changes_the_author_skips() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let labels = ["anchor_campaign_suspend_cse", "cse_reports_embedding_v2"];
        for entity in BOTH {
            for s in [
                AuthorSkips::default(),
                skips(Some(0), Some(false)),
                skips(Some(250), None),
            ] {
                for a in [
                    with_followers(d),
                    with_followers(250),
                    Author {
                        high_cred: true,
                        ..with_followers(1)
                    },
                ] {
                    let plain = first_page_stop(entity, &author_facts(entity, a, &[]), s).unwrap();
                    let cse =
                        first_page_stop(entity, &author_facts(entity, a, &labels), s).unwrap();
                    assert_eq!(plain.map(|c| c.reason), cse.map(|c| c.reason));
                }
            }
        }
    }

    #[test]
    fn lifting_a_skip_is_harsher_and_adding_one_is_not() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as u32;
        let below = (d / 4).max(1);
        let today = AuthorSkips::default();
        assert!(today.harsher_than(today).is_empty());
        assert_eq!(
            skips(Some(0), None).harsher_than(today),
            [format!(
                "skips fewer high-follower accounts ({d}+ to {}+)",
                types::stored_off_follower_threshold()
            )]
        );
        let c = follower_ceiling() as u32;
        assert_eq!(
            skips(Some(c * 3), None).harsher_than(today),
            [
                format!("skips fewer high-follower accounts ({d}+ to {}+)", c * 3),
                format!(
                    "raises the follower ceiling ({c}+ to {}+), acting on accounts every other head skips",
                    c * 3
                ),
            ]
        );
        assert_eq!(
            skips(Some(c), None).harsher_than(today),
            [format!("skips fewer high-follower accounts ({d}+ to {c}+)")]
        );
        assert_eq!(
            skips(Some(d * 5), None).harsher_than(today),
            [format!(
                "skips fewer high-follower accounts ({d}+ to {}+)",
                d * 5
            )]
        );
        assert!(skips(Some(below), None).harsher_than(today).is_empty());
        assert!(skips(Some(d), None).harsher_than(today).is_empty());
        assert!(today.harsher_than(skips(Some(0), None)).is_empty());
        assert!(
            skips(Some(below), None)
                .harsher_than(skips(Some(0), None))
                .is_empty()
        );
        assert_eq!(
            skips(None, Some(false)).harsher_than(today),
            ["stops skipping high-credibility accounts"]
        );
        assert!(
            skips(None, Some(true))
                .harsher_than(skips(None, Some(false)))
                .is_empty()
        );
        assert_eq!(skips(Some(0), Some(false)).harsher_than(today).len(), 2);
    }

    fn policy() -> TestAccountPolicy {
        TestAccountPolicy::new(
            &[" Test.Example.com ".into(), String::new()],
            &["qa_test_role".into(), "test_role".into()],
            &["203.0.113.0/24".into(), "2001:db8:a::/48".into()],
        )
        .unwrap()
    }

    fn account(
        email: Option<&str>,
        state: Option<EmailConfirmedState>,
        ip: Option<&str>,
        roles: &[&str],
        status: Option<&str>,
    ) -> GizmoduckFacts {
        GizmoduckFacts {
            present: true,
            login_email: email.map(str::to_owned),
            email_confirmed: state,
            creation_ip: ip.map(str::to_owned),
            roles: roles.iter().map(|r| (*r).to_owned()).collect(),
            test_user_status: status.map(str::to_owned),
            ..Default::default()
        }
    }

    fn email_signal(email: &str, state: Option<EmailConfirmedState>) -> Option<TestAccountSignal> {
        policy().signal(&account(Some(email), state, None, &[], None))
    }

    #[test]
    fn a_test_status_matches_with_any_policy() {
        for p in [TestAccountPolicy::default(), policy()] {
            for (status, want) in [
                (Some("Temporary"), Some(TestAccountSignal::TestUserStatus)),
                (Some("Permanent"), Some(TestAccountSignal::TestUserStatus)),
                (Some("None"), None),
                (Some("Unknown(7)"), None),
                (Some("temporary"), None),
                (None, None),
            ] {
                assert_eq!(
                    p.signal(&account(None, None, None, &[], status)),
                    want,
                    "{status:?}"
                );
            }
        }
        assert!(TestAccountPolicy::default().is_empty());
    }

    #[test]
    fn a_role_matches_only_exactly() {
        let p = policy();
        let role = |r: &[&str]| p.signal(&account(None, None, None, r, None));
        assert_eq!(role(&["x", "test_role"]), Some(TestAccountSignal::Role));
        assert_eq!(role(&["qa_test_role"]), Some(TestAccountSignal::Role));
        for miss in [
            "qa_test",
            "qa_test_role_x",
            "QA_Test_Role",
            " qa_test_role",
            "",
        ] {
            assert_eq!(role(&[miss]), None, "{miss:?}");
        }
        assert_eq!(role(&[]), None);
    }

    #[test]
    fn only_a_confirmed_address_at_the_exact_domain_matches() {
        use EmailConfirmedState::*;
        let yes = Some(TestAccountSignal::EmailDomain);
        assert_eq!(email_signal("a@test.example.com", Some(Confirmed)), yes);
        assert_eq!(email_signal("A.B+c@TEST.EXAMPLE.COM", Some(Confirmed)), yes);
        assert_eq!(email_signal("  a@test.example.com\t", Some(Confirmed)), yes);
        for state in [Some(Unconfirmed), Some(Excused), Some(Grandfathered), None] {
            assert_eq!(email_signal("a@test.example.com", state), None, "{state:?}");
        }
        for evasion in [
            "a@eviltest.example.com",
            "a@sub.test.example.com",
            "a@test.example.com.evil.test",
            "a@test.example.co",
            "a@test.example.com.",
            "test.example.com@evil.test",
            "a@evil.test@test.example.com",
            "@test.example.com",
            "a@",
            "test.example.com",
            "<redacted>",
            "a@test.ex\u{430}mple.com",
            "a@test.example.com\u{200b}",
            "",
        ] {
            assert_eq!(email_signal(evasion, Some(Confirmed)), None, "{evasion:?}");
        }
        assert_eq!(
            policy().signal(&account(None, Some(Confirmed), None, &[], None)),
            None,
            "no address"
        );
    }

    #[test]
    fn networks_match_at_their_edges_only() {
        let ip = |s: &str| policy().signal(&account(None, None, Some(s), &[], None));
        let yes = Some(TestAccountSignal::CreationCidr);
        for inside in [
            "203.0.113.0",
            "203.0.113.255",
            " 203.0.113.9 ",
            "::ffff:203.0.113.9",
            "[::ffff:203.0.113.9]",
            "2001:db8:a::1",
            "2001:db8:a:ffff:ffff:ffff:ffff:ffff",
            "[2001:db8:a::1]",
        ] {
            assert_eq!(ip(inside), yes, "{inside}");
        }
        for outside in [
            "203.0.112.255",
            "203.0.114.0",
            "2001:db8:9:ffff:ffff:ffff:ffff:ffff",
            "2001:db8:b::",
            "::203.0.113.9",
            "203.0.113",
            "203.0.113.9:443",
            "203.0.113.9/24",
            "not an ip",
            "",
        ] {
            assert_eq!(ip(outside), None, "{outside}");
        }
    }

    #[test]
    fn network_parsing() {
        let net = |s: &str| IpNet::parse(s);
        let v4 = |s: &str| s.parse::<IpAddr>().unwrap();
        assert!(net("0.0.0.0/0").unwrap().contains(v4("198.51.100.1")));
        assert!(!net("0.0.0.0/0").unwrap().contains(v4("2001:db8::1")));
        assert!(net("::/0").unwrap().contains(v4("2001:db8::1")));
        let host = net("198.51.100.7").unwrap();
        assert!(host.contains(v4("198.51.100.7")) && !host.contains(v4("198.51.100.8")));
        assert_eq!(
            net("::ffff:203.0.113.0/120").unwrap(),
            net("203.0.113.0/24").unwrap(),
            "an IPv4-mapped network is the IPv4 network"
        );
        assert!(
            net("203.0.113.77/24").unwrap().contains(v4("203.0.113.1")),
            "host bits in the network address are ignored"
        );
        for bad in [
            "203.0.113.0/33",
            "2001:db8::/129",
            "203.0.113.0/",
            "/24",
            "x/8",
            "203.0.113.0/-1",
        ] {
            assert!(net(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn bad_configuration_is_refused() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        for domain in [
            "@test.example.com",
            "test",
            ".test.example.com",
            "test.example.com.",
            "a b.com",
            "t\u{e9}st.example.com",
        ] {
            assert!(
                TestAccountPolicy::new(&s(&[domain]), &[], &[]).is_err(),
                "{domain}"
            );
        }
        assert!(TestAccountPolicy::new(&[], &[], &s(&["203.0.113.0/40"])).is_err());
        let p = TestAccountPolicy::new(&s(&["", " "]), &s(&[" ", ""]), &s(&[""])).unwrap();
        assert!(p.is_empty(), "blank entries are ignored");
        assert_eq!(policy().sizes(), (1, 2, 2));
    }

    #[test]
    fn signals_are_reported_in_order() {
        let p = policy();
        let all = account(
            Some("a@test.example.com"),
            Some(EmailConfirmedState::Confirmed),
            Some("203.0.113.1"),
            &["qa_test_role"],
            Some("Permanent"),
        );
        assert_eq!(p.signal(&all), Some(TestAccountSignal::TestUserStatus));
        let no_status = GizmoduckFacts {
            test_user_status: None,
            ..all.clone()
        };
        assert_eq!(p.signal(&no_status), Some(TestAccountSignal::Role));
        let no_role = GizmoduckFacts {
            roles: Vec::new(),
            ..no_status
        };
        assert_eq!(p.signal(&no_role), Some(TestAccountSignal::EmailDomain));
        let no_email = GizmoduckFacts {
            login_email: None,
            ..no_role
        };
        assert_eq!(p.signal(&no_email), Some(TestAccountSignal::CreationCidr));
    }

    fn test_account_facts(entity: EntityType, a: Author, user: GizmoduckFacts) -> Facts {
        let mut f = author_facts(entity, a, &[]);
        *f.user_mut() = user;
        f
    }

    fn by_signal(signal: TestAccountSignal) -> GizmoduckFacts {
        GizmoduckFacts {
            present: true,
            test_signal: Some(signal),
            ..Default::default()
        }
    }

    #[test]
    fn the_page_path_stops_test_accounts_after_allowlist_and_not_found() {
        let c = follower_ceiling() as i64;
        let test = GizmoduckFacts {
            present: true,
            test_user_status: Some("Temporary".into()),
            ..Default::default()
        };
        for entity in BOTH {
            for s in [AuthorSkips::default(), skips(Some(0), Some(false))] {
                let stop = |a: Author, user: GizmoduckFacts| {
                    first_page_stop(entity, &test_account_facts(entity, a, user), s)
                        .unwrap()
                        .map(|c| c.id)
                };
                assert_eq!(
                    stop(with_followers(1), test.clone()),
                    Some("gizmoduck_test_user")
                );
                assert_eq!(
                    stop(
                        with_followers(1),
                        by_signal(TestAccountSignal::TestUserStatus)
                    ),
                    Some("gizmoduck_test_user")
                );
                for signal in [
                    TestAccountSignal::Role,
                    TestAccountSignal::EmailDomain,
                    TestAccountSignal::CreationCidr,
                ] {
                    assert_eq!(
                        stop(with_followers(c), by_signal(signal)),
                        Some("test_account"),
                        "{entity:?} {signal:?}: before the ceiling"
                    );
                }
                let allowlisted = Author {
                    allowlisted: true,
                    ..with_followers(1)
                };
                assert_eq!(stop(allowlisted, test.clone()), Some("user_in_allowlist"));
                assert_eq!(
                    stop(
                        with_followers(1),
                        GizmoduckFacts {
                            present: false,
                            ..test.clone()
                        }
                    ),
                    Some("user_not_found")
                );
                assert_eq!(
                    stop(with_followers(1), by_signal_none()),
                    None,
                    "{entity:?} {s:?}"
                );
            }
        }
    }

    fn by_signal_none() -> GizmoduckFacts {
        GizmoduckFacts {
            present: true,
            ..Default::default()
        }
    }

    #[test]
    fn the_rules_path_stops_test_accounts_after_allowlist_and_not_found() {
        let rules = compiled(BAKED_USER);
        let c = follower_ceiling() as i64;
        let decide = |a: Author, user: GizmoduckFacts, labels: &[&str]| {
            let mut f = test_account_facts(EntityType::User, a, user);
            f.score.labels = labels.iter().map(|l| (*l).to_owned()).collect();
            crate::rules::decide_with_id(&rules, &f).unwrap()
        };
        let suspend = ["anchor_campaign_suspend"];
        let (d, id) = decide(with_followers(1), by_signal_none(), &suspend);
        assert!(matches!(d, crate::decision::Decision::Act(_)), "{d:?}");
        assert_eq!(id, "anchor_campaign_suspend");
        for (user, want_id, want_reason) in [
            (
                GizmoduckFacts {
                    present: true,
                    test_user_status: Some("Permanent".into()),
                    ..Default::default()
                },
                "gizmoduck_test_user",
                "test_user_skipped",
            ),
            (
                by_signal(TestAccountSignal::Role),
                "test_account",
                "test_user_skipped",
            ),
        ] {
            for followers in [1, c] {
                let (d, id) = decide(with_followers(followers), user.clone(), &suspend);
                assert_eq!(id, want_id);
                assert_eq!(d, crate::decision::Decision::Skip(want_reason.into()));
            }
            let allowlisted = Author {
                allowlisted: true,
                ..with_followers(1)
            };
            assert_eq!(
                decide(allowlisted, user.clone(), &suspend).1,
                "user_in_allowlist"
            );
        }
    }

    #[test]
    fn invited_replies_stop_post_decisions_last_on_both_paths() {
        let d = DEFAULT_FOLLOWER_SKIP.ceil() as i64;
        let c = follower_ceiling() as i64;
        let with = |entity: EntityType, a: Author, invited: InvitedReply| {
            let mut f = author_facts(entity, a, &["anchor_campaign_suspend"]);
            if let EntityFacts::Post(p) = &mut f.entity {
                p.invited = invited;
            }
            f
        };
        let page = |f: &Facts| {
            first_page_stop(f.entity_type, f, AuthorSkips::default())
                .unwrap()
                .map(|c| c.reason)
        };
        for (invited, want) in [
            (InvitedReply::Mentioned, Some("invited_reply_skipped")),
            (InvitedReply::Unknown, Some("invited_reply_unknown_skipped")),
            (InvitedReply::NotInvited, None),
            (InvitedReply::NotAReply, None),
        ] {
            assert_eq!(
                page(&with(EntityType::Post, with_followers(1), invited)),
                want,
                "{invited:?}"
            );
            for s in [skips(Some(0), Some(false)), skips(Some(250), None)] {
                let f = with(EntityType::Post, with_followers(1), invited);
                assert_eq!(
                    first_page_stop(EntityType::Post, &f, s)
                        .unwrap()
                        .map(|c| c.reason),
                    want,
                    "{invited:?} {s:?}: no head switch"
                );
            }
            assert_eq!(
                page(&with(EntityType::Post, with_followers(d), invited)),
                Some("high_follower_count")
            );
            assert_eq!(
                page(&with(EntityType::Post, with_followers(c), invited)),
                Some("very_high_follower_count")
            );
            let waived = Author {
                waived: true,
                ..with_followers(d)
            };
            assert_eq!(
                page(&with(EntityType::Post, waived, invited)),
                want,
                "{invited:?}: not waived"
            );
        }
        assert_eq!(
            page(&with(
                EntityType::User,
                with_followers(1),
                InvitedReply::Mentioned
            )),
            None
        );

        let rules = compiled(BAKED_POST);
        let decide = |a: Author, invited: InvitedReply| {
            let mut f = with(EntityType::Post, a, invited);
            f.score.labels = vec!["cse_reports_embedding_v2".to_owned()];
            let (d, id) = crate::rules::decide_with_id(&rules, &f).unwrap();
            match d {
                crate::decision::Decision::Act(_) => format!("act:{id}"),
                crate::decision::Decision::Skip(r) => r,
            }
        };
        assert!(decide(with_followers(1), InvitedReply::NotInvited).starts_with("act:"));
        assert_eq!(
            decide(with_followers(1), InvitedReply::Mentioned),
            "invited_reply_skipped"
        );
        assert_eq!(
            decide(with_followers(1), InvitedReply::Unknown),
            "invited_reply_unknown_skipped"
        );
        assert_eq!(
            decide(with_followers(d), InvitedReply::Mentioned),
            "high_follower_count",
            "a skip the rules chose keeps its reason"
        );
    }

                #[test]
    fn test_account_checks_leave_the_comparison_alone() {
        assert!(status(EntityType::User, &compiled(BAKED_USER)).ok);
        assert!(status(EntityType::Post, &compiled(BAKED_POST)).ok);
        for check in [
            GIZMODUCK_TEST_USER,
            TEST_ACCOUNT,
            INVITED_REPLY,
            INVITED_REPLY_UNKNOWN,
        ] {
            let rule = format!(
                "  - id: {}\n    when: user.deactivated\n    then:\n      kind: skip\n      reason: {}\n\n",
                check.id, check.reason
            );
            for before in ["user_in_allowlist", "high_follower_count"] {
                let st = edited_user(|y| {
                    let (s, _) = rule_block(y, before);
                    format!("{}{}{}", &y[..s], rule, &y[s..])
                });
                assert!(st.ok, "{} before {before}: {:?}", check.id, st.problems);
            }
            assert!(
                !USER_CHECKS
                    .iter()
                    .chain(POST_CHECKS)
                    .any(|c| c.id == check.id)
            );
        }
    }
}
