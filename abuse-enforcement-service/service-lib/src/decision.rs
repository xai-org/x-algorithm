#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Skip(String),
    Act(Vec<ActionSpec>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionSpec {
    SuspendUser {
        perm: bool,
        policy: String,
    },
    AddLabelsV2 {
        labels: Vec<String>,
        ttl_msec: Option<i64>,
    },
    AddPostLabelsV2 {
        labels: Vec<String>,
        ttl_msec: Option<i64>,
    },
    Arkose,
    Captcha,
    SpamLivenessCheck,
}

impl ActionSpec {
    pub fn dedup_action_class(&self) -> u8 {
        match self {
            ActionSpec::SuspendUser { policy, .. } if is_cse_policy(policy) => {
                crate::dedup_cache::CLASS_SUSPEND_CSE
            }
            ActionSpec::SuspendUser { perm: true, .. } => {
                crate::dedup_cache::CLASS_SUSPEND_PERMANENT
            }
            ActionSpec::SuspendUser { perm: false, .. } => crate::dedup_cache::CLASS_SUSPEND,
            ActionSpec::Arkose | ActionSpec::Captcha | ActionSpec::SpamLivenessCheck => {
                crate::dedup_cache::CLASS_CHALLENGE
            }
            ActionSpec::AddLabelsV2 { .. } | ActionSpec::AddPostLabelsV2 { .. } => {
                crate::dedup_cache::CLASS_LABEL
            }
        }
    }

    pub fn is_permanent_cse_suspension(&self) -> bool {
        matches!(self, ActionSpec::SuspendUser { perm: true, policy } if is_cse_policy(policy))
    }
}

pub fn is_cse_policy(policy: &str) -> bool {
    policy.eq_ignore_ascii_case(crate::action_control::SuspensionPolicy::Cse.as_str())
}
