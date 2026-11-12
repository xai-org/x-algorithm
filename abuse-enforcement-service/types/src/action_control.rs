use serde::{Deserialize, Serialize};

pub const ACTION_CONTROL_RECORD_VERSION: u32 = 7;
pub const POST_LABEL_RECORD_VERSION: u32 = 7;
pub const SUSPENSION_OPTIONS_RECORD_VERSION: u32 = 6;
pub const AUTHOR_SKIPS_RECORD_VERSION: u32 = 5;
pub const ROLLOUT_RECORD_VERSION: u32 = 4;
pub const ROWS_RECORD_VERSION: u32 = 3;
pub const BAND_RECORD_VERSION: u32 = 1;

pub const ROLLOUT_ALL_BP: u32 = 10_000;

pub fn rollout_bucket(salt: &str, head: &str, author_id: i64) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let id = author_id.to_string();
    let salt_sep: &[u8] = if salt.is_empty() { b"" } else { b"/" };
    for b in salt
        .bytes()
        .chain(salt_sep.iter().copied())
        .chain(head.bytes())
        .chain([b'/'])
        .chain(id.bytes())
    {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    (h % u64::from(ROLLOUT_ALL_BP)) as u32
}

pub fn in_rollout(rollout_bp: Option<u32>, salt: &str, head: &str, author_id: i64) -> bool {
    rollout_bp.is_none_or(|bp| bp >= ROLLOUT_ALL_BP || rollout_bucket(salt, head, author_id) < bp)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceLevel {
    Low,
    Mid,
    High,
}

impl ConfidenceLevel {
    pub const ALL: [ConfidenceLevel; 3] = [
        ConfidenceLevel::Low,
        ConfidenceLevel::Mid,
        ConfidenceLevel::High,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ConfidenceLevel::Low => "low",
            ConfidenceLevel::Mid => "mid",
            ConfidenceLevel::High => "high",
        }
    }

                pub fn floor(self) -> f64 {
        match self {
            ConfidenceLevel::Low => 0.70,
            ConfidenceLevel::Mid => 0.85,
            ConfidenceLevel::High => 0.95,
        }
    }

        pub fn supported_by(lower_bound: f64) -> Option<ConfidenceLevel> {
        Self::ALL
            .into_iter()
            .rev()
            .find(|l| lower_bound >= l.floor())
    }

    pub fn parse(s: &str) -> Option<ConfidenceLevel> {
        Self::ALL.into_iter().find(|l| l.as_str() == s)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum LevelRow {
    Low,
    Mid,
    High,
    Missing,
}

impl LevelRow {
    pub const ALL: [LevelRow; 4] = [
        LevelRow::Low,
        LevelRow::Mid,
        LevelRow::High,
        LevelRow::Missing,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            LevelRow::Low => "low",
            LevelRow::Mid => "mid",
            LevelRow::High => "high",
            LevelRow::Missing => "missing",
        }
    }

        pub fn level(self) -> Option<ConfidenceLevel> {
        match self {
            LevelRow::Low => Some(ConfidenceLevel::Low),
            LevelRow::Mid => Some(ConfidenceLevel::Mid),
            LevelRow::High => Some(ConfidenceLevel::High),
            LevelRow::Missing => None,
        }
    }
}

impl From<ConfidenceLevel> for LevelRow {
    fn from(level: ConfidenceLevel) -> Self {
        match level {
            ConfidenceLevel::Low => LevelRow::Low,
            ConfidenceLevel::Mid => LevelRow::Mid,
            ConfidenceLevel::High => LevelRow::High,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ConfidenceEvidence {
            pub precision_lower_bound: f64,
    pub reads: u32,
    pub humans: u32,
        pub read_at_ms: i64,
        #[serde(default)]
    pub link: String,
}

impl Eq for ConfidenceEvidence {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct EvidenceOverride {
    pub by: String,
    pub reason: String,
    pub expires_at_ms: i64,
}

pub const EVIDENCE_OVERRIDE_MAX_MS: i64 = 7 * 86_400_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
        Rules,
            Recording,
        Live,
            Rows,
}

impl ControlMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ControlMode::Rules => "rules",
            ControlMode::Recording => "recording",
            ControlMode::Live => "live",
            ControlMode::Rows => "rows",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Lifetime {
    #[serde(rename = "7d")]
    Days7,
    #[serde(rename = "30d")]
    Days30,
    #[serde(rename = "permanent")]
    Permanent,
}

impl Lifetime {
        pub fn ttl_msec(self) -> Option<i64> {
        match self {
            Lifetime::Days7 => Some(7 * 86_400_000),
            Lifetime::Days30 => Some(30 * 86_400_000),
            Lifetime::Permanent => None,
        }
    }
}

#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum SuspensionPolicy {
    #[default]
    PlatformManipulation,
    Cse,
}

impl SuspensionPolicy {
    pub const ALL: [SuspensionPolicy; 2] = [
        SuspensionPolicy::PlatformManipulation,
        SuspensionPolicy::Cse,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SuspensionPolicy::PlatformManipulation => "PlatformManipulation",
            SuspensionPolicy::Cse => "Cse",
        }
    }

    fn is_default(&self) -> bool {
        *self == SuspensionPolicy::default()
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

pub const PAGE_POST_LABELS: &[&str] = &[
    "NsfaHighPrecision",
    "NsfaLimitedInventory",
    "NsfwHighPrecision",
    "SoftNsfw",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PostLabelName(&'static str);

impl PostLabelName {
    pub fn parse(name: &str) -> Option<Self> {
        PAGE_POST_LABELS
            .iter()
            .find(|l| **l == name)
            .map(|l| Self(l))
    }

    pub fn all() -> impl Iterator<Item = Self> {
        PAGE_POST_LABELS.iter().map(|l| Self(l))
    }

    pub fn as_str(self) -> &'static str {
        self.0
    }
}

impl Serialize for PostLabelName {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for PostLabelName {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        Self::parse(&name).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "`{name}` is not a label the page may put on a post"
            ))
        })
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PageAction {
        Suspend {
        #[serde(default, skip_serializing_if = "is_false")]
        permanent: bool,
        #[serde(default, skip_serializing_if = "SuspensionPolicy::is_default")]
        policy: SuspensionPolicy,
    },
        LivenessCheck {},
        SpamHighRecall { lifetime: Lifetime },
            PhoenixSkip { lifetime: Lifetime },
        PostSpamHighRecall { lifetime: Lifetime },
            HideReply { lifetime: Lifetime },
        SuspendAuthor {},
        PostLabel {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        label: PostLabelName,
        lifetime: Lifetime,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ControlEntity {
    User,
    Post,
}

impl ControlEntity {
    pub fn as_str(self) -> &'static str {
        match self {
            ControlEntity::User => "user",
            ControlEntity::Post => "post",
        }
    }
}

impl PageAction {
        pub const SUSPEND: PageAction = PageAction::Suspend {
        permanent: false,
        policy: SuspensionPolicy::PlatformManipulation,
    };

        pub fn uses_suspension_options(self) -> bool {
        matches!(self, PageAction::Suspend { permanent, policy } if permanent || !policy.is_default())
    }

    pub fn entity(self) -> ControlEntity {
        match self {
            PageAction::Suspend { .. }
            | PageAction::LivenessCheck {}
            | PageAction::SpamHighRecall { .. }
            | PageAction::PhoenixSkip { .. } => ControlEntity::User,
            PageAction::PostSpamHighRecall { .. }
            | PageAction::HideReply { .. }
            | PageAction::SuspendAuthor {}
            | PageAction::PostLabel { .. } => ControlEntity::Post,
        }
    }

        pub fn is_post_label(self) -> bool {
        matches!(self, PageAction::PostLabel { .. })
    }

                pub fn menu(entity: ControlEntity) -> Vec<PageAction> {
        match entity {
            ControlEntity::User => [false, true]
                .into_iter()
                .flat_map(|permanent| {
                    SuspensionPolicy::ALL
                        .into_iter()
                        .map(move |policy| PageAction::Suspend { permanent, policy })
                })
                .chain([
                    PageAction::LivenessCheck {},
                    PageAction::SpamHighRecall {
                        lifetime: Lifetime::Days7,
                    },
                    PageAction::SpamHighRecall {
                        lifetime: Lifetime::Days30,
                    },
                    PageAction::SpamHighRecall {
                        lifetime: Lifetime::Permanent,
                    },
                    PageAction::PhoenixSkip {
                        lifetime: Lifetime::Days7,
                    },
                    PageAction::PhoenixSkip {
                        lifetime: Lifetime::Days30,
                    },
                    PageAction::PhoenixSkip {
                        lifetime: Lifetime::Permanent,
                    },
                ])
                .collect(),
            ControlEntity::Post => vec![
                PageAction::PostSpamHighRecall {
                    lifetime: Lifetime::Days30,
                },
                PageAction::PostSpamHighRecall {
                    lifetime: Lifetime::Permanent,
                },
                PageAction::HideReply {
                    lifetime: Lifetime::Days7,
                },
                PageAction::HideReply {
                    lifetime: Lifetime::Days30,
                },
                PageAction::HideReply {
                    lifetime: Lifetime::Permanent,
                },
                PageAction::SuspendAuthor {},
            ]
            .into_iter()
            .chain(PostLabelName::all().flat_map(|label| {
                [Lifetime::Days7, Lifetime::Days30, Lifetime::Permanent]
                    .map(|lifetime| PageAction::PostLabel { label, lifetime })
            }))
            .collect(),
        }
    }

            pub fn suggested(entity: ControlEntity, row: LevelRow) -> Vec<PageAction> {
        match (entity, row) {
            (ControlEntity::User, LevelRow::Low) => vec![PageAction::LivenessCheck {}],
            (ControlEntity::User, LevelRow::Mid) => vec![
                PageAction::LivenessCheck {},
                PageAction::SpamHighRecall {
                    lifetime: Lifetime::Days7,
                },
            ],
            (ControlEntity::User, LevelRow::High) => vec![PageAction::SUSPEND],
            (ControlEntity::Post, LevelRow::Mid) => vec![PageAction::PostSpamHighRecall {
                lifetime: Lifetime::Days30,
            }],
            (ControlEntity::Post, LevelRow::High) => vec![PageAction::PostSpamHighRecall {
                lifetime: Lifetime::Permanent,
            }],
            (ControlEntity::Post, LevelRow::Low) | (_, LevelRow::Missing) => vec![],
        }
    }

                pub fn kind(self) -> &'static str {
        match self {
            PageAction::Suspend { .. } => "suspend",
            PageAction::LivenessCheck {} => "liveness_check",
            PageAction::SpamHighRecall { .. } => "spam_high_recall",
            PageAction::PhoenixSkip { .. } => "phoenix_skip",
            PageAction::PostSpamHighRecall { .. } => "post_spam_high_recall",
            PageAction::HideReply { .. } => "hide_reply",
            PageAction::SuspendAuthor {} => "suspend_author",
            PageAction::PostLabel { label, .. } => label.as_str(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlRecord {
    pub v: u32,
    pub entity_type: ControlEntity,
    pub head: String,
    #[serde(default)]
    pub classifier: Option<String>,
    #[serde(default)]
    pub band: Option<String>,
    pub mode: ControlMode,
    #[serde(default)]
    pub actions: Vec<PageAction>,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub updated_by: String,
    #[serde(default)]
    pub key_id: String,
    #[serde(default)]
    pub updated_at_ms: i64,
        pub version: u64,
        #[serde(default)]
    pub write_nonce: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<LevelRow>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Box<ConfidenceEvidence>>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_override: Option<Box<EvidenceOverride>>,
                    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub admit_unlisted_labels: bool,
                        #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_bp: Option<u32>,
                    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follower_threshold: Option<u32>,
                #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_high_credibility: Option<bool>,
}

impl ActionControlRecord {
            pub const FIELDS: &'static [&'static str] = &[
        "v",
        "entity_type",
        "head",
        "classifier",
        "band",
        "mode",
        "actions",
        "reason",
        "updated_by",
        "key_id",
        "updated_at_ms",
        "version",
        "write_nonce",
        "confidence",
        "evidence",
        "evidence_override",
        "admit_unlisted_labels",
        "rollout_bp",
        "follower_threshold",
        "skip_high_credibility",
    ];

            pub fn uses_confidence(&self) -> bool {
        self.confidence.is_some()
            || self.evidence.is_some()
            || self.evidence_override.is_some()
            || self.mode == ControlMode::Rows
    }

            pub fn uses_suspension_options(&self) -> bool {
        self.actions.iter().any(|a| a.uses_suspension_options())
    }

        pub fn sets_author_skips(&self) -> bool {
        self.follower_threshold.is_some() || self.skip_high_credibility.is_some()
    }

        pub fn uses_post_label(&self) -> bool {
        self.actions.iter().any(|a| a.is_post_label())
    }

                            pub fn schema_version(&self) -> u32 {
        if self.uses_post_label() {
            POST_LABEL_RECORD_VERSION
        } else if self.uses_suspension_options() {
            SUSPENSION_OPTIONS_RECORD_VERSION
        } else if self.sets_author_skips() {
            AUTHOR_SKIPS_RECORD_VERSION
        } else if self.rollout_bp.is_some() {
            ROLLOUT_RECORD_VERSION
        } else if self.uses_confidence() {
            ROWS_RECORD_VERSION
        } else {
            BAND_RECORD_VERSION
        }
    }

                pub fn rollout_takes(&self, salt: &str, author_id: i64) -> bool {
        in_rollout(self.rollout_bp, salt, &self.head, author_id)
    }

            pub fn parse_strict(bytes: &[u8]) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if let Some(obj) = value.as_object()
            && let Some(k) = obj.keys().find(|k| !Self::FIELDS.contains(&k.as_str()))
        {
            return Err(format!("unknown field `{k}`"));
        }
        serde_json::from_value(value).map_err(|e| e.to_string())
    }

            pub fn scope(&self) -> &'static str {
        match (
            self.classifier.is_some(),
            self.band.is_some(),
            self.confidence.is_some(),
        ) {
            (false, _, true) => "confidence",
            (true, _, true) => "classifier+confidence",
            (false, false, false) => "head",
            (true, false, false) => "classifier",
            (false, true, false) => "band",
            (true, true, false) => "classifier+band",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlView {
    #[serde(flatten)]
    pub record: ActionControlRecord,
    pub scope: String,
    pub lkey: String,
        pub effective_mode: ControlMode,
            #[serde(default)]
    pub forced_recording_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct ActionControlPutRequest {
            pub entity_type: ControlEntity,
    pub head: String,
    #[serde(default)]
    pub classifier: Option<String>,
    #[serde(default)]
    pub band: Option<String>,
    pub mode: ControlMode,
    #[serde(default)]
    pub actions: Vec<PageAction>,
    #[serde(default)]
    pub reason: String,
        #[serde(default)]
    pub expected_version: Option<u64>,
        #[serde(default)]
    pub confirm: Option<String>,
        #[serde(default)]
    pub updated_by: Option<String>,
        #[serde(default)]
    pub confidence: Option<LevelRow>,
        #[serde(default)]
    pub evidence: Option<ConfidenceEvidence>,
            #[serde(default)]
    pub evidence_override: Option<EvidenceOverride>,
            #[serde(default)]
    pub rows: Option<RowsRequest>,
            #[serde(default)]
    pub admit_unlisted_labels: Option<bool>,
            #[serde(default)]
    pub rollout_bp: Option<u32>,
            #[serde(default, skip_serializing_if = "Option::is_none", with = "keep_or_set")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<u32>, nullable))]
    pub follower_threshold: Option<Option<u32>>,
            #[serde(default, skip_serializing_if = "Option::is_none", with = "keep_or_set")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<bool>, nullable))]
    pub skip_high_credibility: Option<Option<bool>>,
}

mod keep_or_set {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<T: Serialize, S: Serializer>(
        v: &Option<Option<T>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        v.as_ref().and_then(Option::as_ref).serialize(s)
    }

    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Option<T>>, D::Error> {
        Option::<T>::deserialize(d).map(Some)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FollowerSkipView {
            pub today_threshold: u64,
            pub ceiling: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct RowsRequest {
    #[serde(default)]
    pub low: Option<RowSetting>,
    #[serde(default)]
    pub mid: Option<RowSetting>,
    #[serde(default)]
    pub high: Option<RowSetting>,
    #[serde(default)]
    pub missing: Option<RowSetting>,
}

impl RowsRequest {
    pub fn get(&self, row: LevelRow) -> Option<&RowSetting> {
        match row {
            LevelRow::Low => self.low.as_ref(),
            LevelRow::Mid => self.mid.as_ref(),
            LevelRow::High => self.high.as_ref(),
            LevelRow::Missing => self.missing.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(deny_unknown_fields)]
pub struct RowSetting {
        pub mode: ControlMode,
    #[serde(default)]
    pub actions: Vec<PageAction>,
    #[serde(default)]
    pub evidence: Option<ConfidenceEvidence>,
        #[serde(default)]
    pub evidence_override: Option<EvidenceOverride>,
        #[serde(default)]
    pub expected_version: Option<u64>,
        #[serde(default)]
    pub rollout_bp: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlSeenRow {
    pub entity_type: ControlEntity,
    pub head: String,
        pub classifier: String,
    pub row: LevelRow,
    pub last_seen_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlSeen {
    pub head: String,
    pub classifier: String,
        pub band: String,
    pub entity_type: ControlEntity,
            #[serde(default)]
    pub factory: bool,
    pub last_seen_ms: i64,
        #[serde(default)]
    pub rule_id: String,
            #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub action_specs: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SafetyCheckView {
    pub id: String,
    pub reason: String,
    pub when: String,
        pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SafetyChecksStatus {
    pub ok: bool,
    pub problems: Vec<String>,
    pub checks: Vec<SafetyCheckView>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SafetyChecksByEntity {
    pub user: SafetyChecksStatus,
    pub post: SafetyChecksStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct AllowedActions {
    pub user: Vec<PageAction>,
    pub post: Vec<PageAction>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CseLists {
    pub heads: Vec<String>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct UnreadableRecord {
    pub lkey: String,
    pub error: String,
            #[serde(default)]
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlListResponse {
    pub feature_version: u32,
    pub env: String,
    pub pod: String,
                    pub enabled: bool,
        pub ignore_store: bool,
        pub source: String,
    #[serde(default)]
    pub error: Option<String>,
        pub snapshot_age_ms: i64,
    pub stale: bool,
    pub stale_after_ms: i64,
    pub records: Vec<ActionControlView>,
    pub unreadable: Vec<UnreadableRecord>,
    pub seen: Vec<ActionControlSeen>,
    #[serde(default)]
    pub seen_error: Option<String>,
        #[serde(default)]
    pub seen_rows: Vec<ActionControlSeenRow>,
    pub safety_checks: SafetyChecksByEntity,
    pub allowed_actions: AllowedActions,
    #[serde(default)]
    pub cse: CseLists,
    #[serde(default)]
    pub confidence: Option<ConfidenceView>,
        #[serde(default)]
    pub follower_skip: Option<FollowerSkipView>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RowView {
    pub row: LevelRow,
            pub floor: Option<f64>,
        pub suggested_actions: AllowedActions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ConfidenceView {
    pub rows: Vec<RowView>,
        pub evidence_max_age_ms: i64,
        #[serde(default)]
    pub factory_evidence_max_age_ms: i64,
            #[serde(default)]
    pub config_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlHistoryEntry {
    pub at_ms: i64,
    pub version: u64,
    pub lkey: String,
    pub scope: String,
    pub updated_by: String,
    pub key_id: String,
    pub reason: String,
    pub env: String,
    pub before: Option<ActionControlRecord>,
    pub after: ActionControlRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ActionControlHistoryResponse {
    pub head: String,
    pub entries: Vec<ActionControlHistoryEntry>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PodView {
    pub pod: String,
    pub env: String,
    pub image: String,
    pub action_control: PodActionControlView,
    pub off_switch: PodOffSwitchView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PodActionControlView {
    pub loaded: bool,
    pub snapshot_age_ms: i64,
    pub stale: bool,
    pub records: usize,
        pub page_managed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PodOffSwitchView {
    pub snapshot_age_ms: i64,
        pub active: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_actions_parse_from_the_documented_json() {
        let a: Vec<PageAction> = serde_json::from_str(
            r#"[{"kind":"suspend"},{"kind":"liveness_check"},
                {"kind":"spam_high_recall","lifetime":"7d"},
                {"kind":"post_spam_high_recall","lifetime":"permanent"},
                {"kind":"suspend_author"}]"#,
        )
        .unwrap();
        assert_eq!(
            a,
            vec![
                PageAction::SUSPEND,
                PageAction::LivenessCheck {},
                PageAction::SpamHighRecall {
                    lifetime: Lifetime::Days7
                },
                PageAction::PostSpamHighRecall {
                    lifetime: Lifetime::Permanent
                },
                PageAction::SuspendAuthor {},
            ]
        );
    }

    #[test]
    fn hide_reply_is_a_post_label_with_every_lifetime() {
        let a: Vec<PageAction> = serde_json::from_str(
            r#"[{"kind":"hide_reply","lifetime":"7d"},
                {"kind":"hide_reply","lifetime":"30d"},
                {"kind":"hide_reply","lifetime":"permanent"}]"#,
        )
        .unwrap();
        assert_eq!(a.len(), 3);
        for x in &a {
            assert_eq!(x.entity(), ControlEntity::Post);
            assert_eq!(x.kind(), "hide_reply");
            assert!(PageAction::menu(ControlEntity::Post).contains(x));
            assert!(!PageAction::menu(ControlEntity::User).contains(x));
        }
        assert_ne!(
            a[2].kind(),
            PageAction::PostSpamHighRecall {
                lifetime: Lifetime::Permanent
            }
            .kind(),
            "a row may hide the reply and label it SpamHighRecall at once"
        );
        assert!(serde_json::from_str::<PageAction>(r#"{"kind":"hide_reply"}"#).is_err());
    }

    #[test]
    fn a_suspend_writes_its_options_only_when_they_differ_from_the_default() {
        let plain: PageAction = serde_json::from_str(r#"{"kind":"suspend"}"#).unwrap();
        assert_eq!(plain, PageAction::SUSPEND);
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            r#"{"kind":"suspend"}"#
        );
        let spelled: PageAction = serde_json::from_str(
            r#"{"kind":"suspend","permanent":false,"policy":"PlatformManipulation"}"#,
        )
        .unwrap();
        assert_eq!(spelled, PageAction::SUSPEND);
        assert!(!spelled.uses_suspension_options());
        let perm: PageAction =
            serde_json::from_str(r#"{"kind":"suspend","permanent":true,"policy":"Cse"}"#).unwrap();
        assert_eq!(
            perm,
            PageAction::Suspend {
                permanent: true,
                policy: SuspensionPolicy::Cse
            }
        );
        assert_eq!(
            serde_json::to_string(&perm).unwrap(),
            r#"{"kind":"suspend","permanent":true,"policy":"Cse"}"#
        );
        assert!(perm.uses_suspension_options());
        assert_eq!(perm.kind(), "suspend");
        assert_eq!(perm.entity(), ControlEntity::User);
    }

    #[test]
    fn the_user_menu_offers_each_suspension_once_and_the_post_menu_none() {
        let user = PageAction::menu(ControlEntity::User);
        let suspends: Vec<&PageAction> = user.iter().filter(|a| a.kind() == "suspend").collect();
        assert_eq!(suspends.len(), 2 * SuspensionPolicy::ALL.len());
        assert_eq!(suspends[0], &PageAction::SUSPEND);
        let mut seen = std::collections::HashSet::new();
        assert!(suspends.iter().all(|a| seen.insert(**a)));
        assert!(
            PageAction::menu(ControlEntity::Post)
                .iter()
                .all(|a| !a.uses_suspension_options())
        );
    }

    #[test]
    fn a_record_with_a_suspension_option_is_refused_by_older_images() {
        let old = r#"{"v":3,"entity_type":"user","head":"H","mode":"live","actions":[{"kind":"suspend"}],"version":1,"confidence":"high","rollout_bp":1000}"#;
        let mut r = ActionControlRecord::parse_strict(old.as_bytes()).unwrap();
        assert_eq!(
            r.schema_version(),
            ROLLOUT_RECORD_VERSION,
            "a plain suspend keeps its version"
        );
        r.actions = vec![PageAction::Suspend {
            permanent: true,
            policy: SuspensionPolicy::PlatformManipulation,
        }];
        assert_eq!(r.schema_version(), 6);
        assert_eq!(r.schema_version(), SUSPENSION_OPTIONS_RECORD_VERSION);
        r.actions = vec![PageAction::Suspend {
            permanent: false,
            policy: SuspensionPolicy::Cse,
        }];
        assert_eq!(r.schema_version(), SUSPENSION_OPTIONS_RECORD_VERSION);
        assert!(r.schema_version() > AUTHOR_SKIPS_RECORD_VERSION);
        let value = serde_json::to_value(r.actions[0]).unwrap();
        let keys: Vec<&String> = value.as_object().unwrap().keys().collect();
        assert!(
            keys.iter().any(|k| *k == "policy"),
            "an older image's deny_unknown_fields refuses the action itself too"
        );
    }

    #[test]
    fn actions_outside_the_menu_do_not_parse() {
        for bad in [
            r#"{"kind":"suspend","perm":true}"#,
            r#"{"kind":"suspend","policy":"Spam"}"#,
            r#"{"kind":"suspend","permanent":"yes"}"#,
            r#"{"kind":"suspend_author","permanent":true}"#,
            r#"{"kind":"spam_high_recall","lifetime":"90d"}"#,
            r#"{"kind":"label","labels":["SpamHighRecall"]}"#,
            r#"{"kind":"arkose"}"#,
        ] {
            assert!(serde_json::from_str::<PageAction>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn record_with_an_unknown_field_is_unreadable() {
        let ok = r#"{"v":1,"entity_type":"user","head":"H","mode":"recording","version":1}"#;
        let r: ActionControlRecord = serde_json::from_str(ok).unwrap();
        assert_eq!(r.scope(), "head");
        assert!(r.actions.is_empty());
        let newer = r#"{"v":2,"entity_type":"user","head":"H","mode":"recording","version":1,"new_field":true}"#;
        assert!(ActionControlRecord::parse_strict(newer.as_bytes()).is_err());
        let retired = r#"{"v":1,"entity_type":"user","head":"H","mode":"recording","version":1,"dedup_strength":"suspend"}"#;
        assert!(ActionControlRecord::parse_strict(retired.as_bytes()).is_err());
    }

    #[test]
    fn fields_list_matches_the_record_and_the_view_round_trips() {
        let r = ActionControlRecord::parse_strict(
            br#"{"v":2,"entity_type":"post","head":"H","classifier":"c","band":"b","mode":"live",
                "actions":[{"kind":"suspend_author"}],"reason":"r","updated_by":"u","key_id":"k",
                "updated_at_ms":5,"version":2,"write_nonce":9,"confidence":"high",
                "evidence":{"precision_lower_bound":0.95,"reads":59,"humans":0,"read_at_ms":4,"link":"l"},
                "evidence_override":{"by":"operator@example.com","reason":"seeded","expires_at_ms":6},
                "admit_unlisted_labels":true,"rollout_bp":500,"follower_threshold":250,
                "skip_high_credibility":false}"#,
        )
        .unwrap();
        let keys: Vec<String> = serde_json::to_value(&r)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let mut want: Vec<&str> = ActionControlRecord::FIELDS.to_vec();
        want.sort_unstable();
        let mut got: Vec<&str> = keys.iter().map(String::as_str).collect();
        got.sort_unstable();
        assert_eq!(got, want);
        let view = ActionControlView {
            record: r,
            scope: "classifier+band".into(),
            lkey: "post/cb/H/c/b".into(),
            effective_mode: ControlMode::Recording,
            forced_recording_reason: Some("kill_switch".into()),
        };
        let back: ActionControlView =
            serde_json::from_str(&serde_json::to_string(&view).unwrap()).unwrap();
        assert_eq!(back, view);
    }

    #[test]
    fn record_and_put_need_the_entity() {
        let no_entity = r#"{"v":1,"head":"H","mode":"recording","version":1}"#;
        assert!(serde_json::from_str::<ActionControlRecord>(no_entity).is_err());
        let put = r#"{"head":"H","mode":"recording","reason":"r"}"#;
        assert!(serde_json::from_str::<ActionControlPutRequest>(put).is_err());
        let put = r#"{"entity_type":"post","head":"H","mode":"recording","reason":"r"}"#;
        let p: ActionControlPutRequest = serde_json::from_str(put).unwrap();
        assert_eq!(p.entity_type, ControlEntity::Post);
        let retired = r#"{"entity_type":"post","head":"H","mode":"recording","reason":"r","dedup_strength":"auto"}"#;
        assert!(serde_json::from_str::<ActionControlPutRequest>(retired).is_err());
    }

    #[test]
    fn scopes() {
        let mut r: ActionControlRecord = serde_json::from_str(
            r#"{"v":1,"entity_type":"user","head":"H","mode":"rules","version":1}"#,
        )
        .unwrap();
        r.classifier = Some("c".into());
        assert_eq!(r.scope(), "classifier");
        r.band = Some("b".into());
        assert_eq!(r.scope(), "classifier+band");
        r.classifier = None;
        assert_eq!(r.scope(), "band");
        r.band = None;
        r.confidence = Some(LevelRow::Mid);
        assert_eq!(r.scope(), "confidence");
        r.classifier = Some("c".into());
        assert_eq!(r.scope(), "classifier+confidence");
    }

    #[test]
    fn a_band_record_keeps_the_old_schema_version_and_bytes() {
        let json = r#"{"v":1,"entity_type":"user","head":"H","mode":"live","actions":[{"kind":"suspend"}],"reason":"","updated_by":"","key_id":"","updated_at_ms":0,"version":1,"write_nonce":0}"#;
        let r: ActionControlRecord = serde_json::from_str(json).unwrap();
        assert_eq!(r.schema_version(), BAND_RECORD_VERSION);
        let back = serde_json::to_string(&r).unwrap();
        assert!(!back.contains("confidence"), "{back}");
        assert!(!back.contains("evidence"), "{back}");
        let mut l = r.clone();
        l.confidence = Some(LevelRow::High);
        assert_eq!(
            l.schema_version(),
            3,
            "a row is refused by images that read v: 2 level records"
        );
        l.confidence = Some(LevelRow::Missing);
        assert_eq!(l.schema_version(), ROWS_RECORD_VERSION);
        let mut d = r;
        d.mode = ControlMode::Rows;
        assert_eq!(d.schema_version(), ROWS_RECORD_VERSION);
        l.rollout_bp = Some(100);
        assert_eq!(
            l.schema_version(),
            4,
            "a row with a rollout is refused by images that read v: 3 rows"
        );
        assert_eq!(l.schema_version(), ROLLOUT_RECORD_VERSION);
    }

    #[test]
    fn a_head_record_with_author_skips_is_refused_by_older_images() {
        let old = r#"{"v":1,"entity_type":"user","head":"H","classifier":null,"band":null,"mode":"live","actions":[{"kind":"suspend"}],"reason":"","updated_by":"","key_id":"","updated_at_ms":0,"version":1,"write_nonce":0}"#;
        let r = ActionControlRecord::parse_strict(old.as_bytes()).unwrap();
        assert_eq!(
            (r.follower_threshold, r.skip_high_credibility),
            (None, None)
        );
        assert!(!r.sets_author_skips());
        assert_eq!(serde_json::to_string(&r).unwrap(), old, "bytes unchanged");
        assert_eq!(r.schema_version(), BAND_RECORD_VERSION);
        for (threshold, credibility) in [(Some(0), None), (Some(250), None), (None, Some(false))] {
            let mut h = r.clone();
            h.follower_threshold = threshold;
            h.skip_high_credibility = credibility;
            assert_eq!(h.schema_version(), 5);
            assert_eq!(h.schema_version(), AUTHOR_SKIPS_RECORD_VERSION);
            h.v = h.schema_version();
            let json = serde_json::to_string(&h).unwrap();
            let back = ActionControlRecord::parse_strict(json.as_bytes()).unwrap();
            assert_eq!(back, h);
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            let older: Vec<&String> = value
                .as_object()
                .unwrap()
                .keys()
                .filter(|k| *k == "follower_threshold" || *k == "skip_high_credibility")
                .collect();
            assert_eq!(older.len(), 1, "an older FIELDS list refuses {json}");
            assert!(
                h.v > ROLLOUT_RECORD_VERSION,
                "and so does an older version bound"
            );
        }
    }

    #[test]
    fn a_put_tells_left_out_from_null_from_a_value() {
        let base = r#""entity_type":"post","head":"H","mode":"live","reason":"r""#;
        let parse = |extra: &str| {
            serde_json::from_str::<ActionControlPutRequest>(&format!("{{{base}{extra}}}"))
        };
        let p = parse("").unwrap();
        assert_eq!(
            (p.follower_threshold, p.skip_high_credibility),
            (None, None)
        );
        let p = parse(r#","follower_threshold":null,"skip_high_credibility":null"#).unwrap();
        assert_eq!(
            (p.follower_threshold, p.skip_high_credibility),
            (Some(None), Some(None))
        );
        let p = parse(r#","follower_threshold":0,"skip_high_credibility":false"#).unwrap();
        assert_eq!(
            (p.follower_threshold, p.skip_high_credibility),
            (Some(Some(0)), Some(Some(false)))
        );
        let p = parse(r#","follower_threshold":2500"#).unwrap();
        assert_eq!(p.follower_threshold, Some(Some(2500)));
        for bad in [
            r#","follower_threshold":-1"#,
            r#","follower_threshold":1.5"#,
            r#","follower_threshold":"off""#,
            r#","skip_high_credibility":"no""#,
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        let left_out = serde_json::to_value(parse("").unwrap()).unwrap();
        assert!(left_out.get("follower_threshold").is_none());
        let cleared =
            serde_json::to_value(parse(r#","follower_threshold":null"#).unwrap()).unwrap();
        assert_eq!(
            cleared.get("follower_threshold"),
            Some(&serde_json::Value::Null)
        );
        let set = serde_json::to_value(parse(r#","follower_threshold":0"#).unwrap()).unwrap();
        assert_eq!(set.get("follower_threshold"), Some(&serde_json::json!(0)));
    }

    #[test]
    fn a_row_without_a_rollout_keeps_its_bytes_and_takes_every_account() {
        let json = r#"{"v":3,"entity_type":"post","head":"H","classifier":null,"band":null,"mode":"live","actions":[{"kind":"hide_reply","lifetime":"permanent"}],"reason":"","updated_by":"","key_id":"","updated_at_ms":0,"version":1,"write_nonce":0,"confidence":"high"}"#;
        let r = ActionControlRecord::parse_strict(json.as_bytes()).unwrap();
        assert_eq!(r.rollout_bp, None);
        assert_eq!(serde_json::to_string(&r).unwrap(), json);
        assert_eq!(r.schema_version(), ROWS_RECORD_VERSION);
        assert!((0..10_000).all(|id| r.rollout_takes("", id) && r.rollout_takes("s", id)));
    }

    #[test]
    fn an_image_before_rollouts_refuses_a_row_with_one() {
        let new = r#"{"v":4,"entity_type":"post","head":"H","mode":"live","actions":[{"kind":"hide_reply","lifetime":"permanent"}],"version":2,"confidence":"high","rollout_bp":100}"#;
        let mut fields: Vec<&str> = ActionControlRecord::FIELDS.to_vec();
        fields.retain(|f| *f != "rollout_bp");
        let value: serde_json::Value = serde_json::from_str(new).unwrap();
        let unknown: Vec<&String> = value
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| !fields.contains(&k.as_str()))
            .collect();
        assert_eq!(unknown, ["rollout_bp"], "an older FIELDS list refuses it");
        let r = ActionControlRecord::parse_strict(new.as_bytes()).unwrap();
        assert!(
            r.v > ROWS_RECORD_VERSION,
            "and so does an older version bound"
        );
    }

    #[test]
    fn rollout_buckets_are_pinned() {
        assert_eq!(rollout_bucket("", "ReplyHead", 0), 5_681);
        assert_eq!(rollout_bucket("", "ReplyHead", 12), 6_175);
        assert_eq!(
            rollout_bucket("", "ReplyHead", 1_234_567_890_123_456_789),
            1_110
        );
        assert_eq!(rollout_bucket("", "H", -1), 4_141);
        assert_eq!(rollout_bucket("test-salt", "ReplyHead", 0), 721);
        assert_eq!(rollout_bucket("test-salt", "ReplyHead", 12), 7_891);
        assert_eq!(rollout_bucket("test-salt", "H", -1), 6_715);
        for salt in ["", "test-salt"] {
            assert!((0..1_000).all(|id| rollout_bucket(salt, "H", id) < ROLLOUT_ALL_BP));
        }
    }

            #[test]
    fn an_empty_salt_keeps_the_unsalted_buckets() {
        fn unsalted(head: &str, author_id: i64) -> u32 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in format!("{head}/{author_id}").bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            h ^= h >> 33;
            (h % u64::from(ROLLOUT_ALL_BP)) as u32
        }
        for head in ["ReplyHead", "H", "A"] {
            for id in (-1_000..1_000).chain([i64::MIN, i64::MAX, 1_234_567_890_123_456_789]) {
                assert_eq!(
                    rollout_bucket("", head, id),
                    unsalted(head, id),
                    "{head}/{id}"
                );
            }
        }
    }

            #[test]
    fn a_salt_redraws_every_rollout() {
        let both = (0..100_000i64)
            .filter(|&id| {
                in_rollout(Some(5_000), "", "ReplyHead", id)
                    && in_rollout(Some(5_000), "test-salt", "ReplyHead", id)
            })
            .count();
        assert!(
            (23_500..26_500).contains(&both),
            "{both} of 100,000 in both 50% rollouts"
        );
    }

    #[test]
    fn a_larger_rollout_keeps_every_account_of_a_smaller_one() {
        let steps = [0, 100, 500, 3_000, 5_000, 10_000];
        let n = 200_000i64;
        for salt in ["", "test-salt"] {
            let mut inside_before: Vec<bool> = vec![false; n as usize];
            for bp in steps {
                let mut count = 0usize;
                for id in 0..n {
                    let inside = in_rollout(Some(bp), salt, "ReplyHead", id * 7_919 + 1);
                    assert!(
                        inside || !inside_before[id as usize],
                        "account {id} left at {bp} bp (salt {salt:?})"
                    );
                    inside_before[id as usize] = inside;
                    count += usize::from(inside);
                }
                let share = count as f64 / n as f64;
                let want = f64::from(bp) / 10_000.0;
                assert!(
                    (share - want).abs() < 0.004,
                    "{bp} bp took {share} (salt {salt:?})"
                );
            }
        }
        assert!(
            (0..n).all(|id| !in_rollout(Some(0), "", "H", id)),
            "0 takes nobody"
        );
        assert!(
            (0..n).all(|id| in_rollout(None, "", "H", id)),
            "absent takes everybody"
        );
    }

    #[test]
    fn heads_draw_their_rollouts_apart() {
        let both = (0..100_000i64)
            .filter(|&id| {
                in_rollout(Some(1_000), "", "A", id) && in_rollout(Some(1_000), "", "B", id)
            })
            .count();
        assert!(
            (500..1_500).contains(&both),
            "{both} of 100,000 in both 10% rollouts"
        );
    }

    #[test]
    fn a_retired_mode_or_an_unknown_row_does_not_parse() {
        for bad in [
            r#"{"v":2,"entity_type":"user","head":"H","mode":"confidence_defaults","version":1}"#,
            r#"{"v":3,"entity_type":"user","head":"H","mode":"live","confidence":"very_high","version":1}"#,
        ] {
            assert!(
                ActionControlRecord::parse_strict(bad.as_bytes()).is_err(),
                "{bad}"
            );
        }
        let missing = r#"{"v":3,"entity_type":"user","head":"H","mode":"recording","confidence":"missing","version":1}"#;
        let r = ActionControlRecord::parse_strict(missing.as_bytes()).unwrap();
        assert_eq!(r.confidence, Some(LevelRow::Missing));
        assert_eq!(LevelRow::Missing.level(), None);
        assert_eq!(LevelRow::from(ConfidenceLevel::Mid), LevelRow::Mid);
    }

    #[test]
    fn floors_and_the_level_evidence_supports() {
        assert_eq!(
            ConfidenceLevel::supported_by(0.951),
            Some(ConfidenceLevel::High)
        );
        assert_eq!(
            ConfidenceLevel::supported_by(0.95),
            Some(ConfidenceLevel::High)
        );
        assert_eq!(
            ConfidenceLevel::supported_by(0.9499),
            Some(ConfidenceLevel::Mid)
        );
        assert_eq!(
            ConfidenceLevel::supported_by(0.85),
            Some(ConfidenceLevel::Mid)
        );
        assert_eq!(
            ConfidenceLevel::supported_by(0.70),
            Some(ConfidenceLevel::Low)
        );
        assert_eq!(ConfidenceLevel::supported_by(0.6999), None);
        assert_eq!(ConfidenceLevel::supported_by(f64::NAN), None);
    }

    #[test]
    fn suggested_table() {
        use LevelRow::*;
        let user = |l| PageAction::suggested(ControlEntity::User, l);
        let post = |l| PageAction::suggested(ControlEntity::Post, l);
        assert!(user(Missing).is_empty() && post(Missing).is_empty());
        assert_eq!(user(Low), vec![PageAction::LivenessCheck {}]);
        assert_eq!(
            user(Mid),
            vec![
                PageAction::LivenessCheck {},
                PageAction::SpamHighRecall {
                    lifetime: Lifetime::Days7
                }
            ]
        );
        assert_eq!(user(High), vec![PageAction::SUSPEND]);
        assert!(post(Low).is_empty());
        for l in LevelRow::ALL {
            for (e, a) in [
                (ControlEntity::User, user(l)),
                (ControlEntity::Post, post(l)),
            ] {
                assert!(a.iter().all(|x| x.entity() == e));
                assert!(a.iter().all(|x| PageAction::menu(e).contains(x)), "{l:?}");
            }
        }
        assert!(
            !post(High).contains(&PageAction::SuspendAuthor {}),
            "a post default never suspends the author"
        );
    }

    #[test]
    fn a_confidence_view_from_an_older_image_reads_without_the_factory_age() {
        let view: ConfidenceView =
            serde_json::from_value(serde_json::json!({"rows": [], "evidence_max_age_ms": 7}))
                .unwrap();
        assert_eq!(
            (view.evidence_max_age_ms, view.factory_evidence_max_age_ms),
            (7, 0)
        );
        let json = serde_json::to_value(ConfidenceView {
            factory_evidence_max_age_ms: 9,
            ..view
        })
        .unwrap();
        assert_eq!(json["factory_evidence_max_age_ms"], 9);
    }

    #[test]
    fn a_post_label_names_a_listed_label_and_round_trips() {
        let json = r#"{"kind":"post_label","label":"NsfwHighPrecision","lifetime":"permanent"}"#;
        let a: PageAction = serde_json::from_str(json).unwrap();
        let label = PostLabelName::parse("NsfwHighPrecision").unwrap();
        assert_eq!(
            a,
            PageAction::PostLabel {
                label,
                lifetime: Lifetime::Permanent
            }
        );
        assert_eq!(serde_json::to_string(&a).unwrap(), json);
        assert_eq!(a.entity(), ControlEntity::Post);
        assert_eq!(a.kind(), "NsfwHighPrecision");
        assert!(a.is_post_label());
        let soft = r#"{"kind":"post_label","label":"SoftNsfw","lifetime":"30d"}"#;
        let a: PageAction = serde_json::from_str(soft).unwrap();
        assert_eq!(a.kind(), "SoftNsfw");
        assert_eq!(serde_json::to_string(&a).unwrap(), soft);
        for nsfa in ["NsfaHighPrecision", "NsfaLimitedInventory"] {
            let json = format!(r#"{{"kind":"post_label","label":"{nsfa}","lifetime":"7d"}}"#);
            let a: PageAction = serde_json::from_str(&json).unwrap();
            assert_eq!(a.kind(), nsfa);
            assert_eq!(a.entity(), ControlEntity::Post);
            assert_eq!(serde_json::to_string(&a).unwrap(), json);
        }
        for bad in [
            r#"{"kind":"post_label","label":"softnsfw","lifetime":"permanent"}"#,
            r#"{"kind":"post_label","label":"NsfwHighRecall","lifetime":"permanent"}"#,
            r#"{"kind":"post_label","label":"NsfaHighRecall","lifetime":"permanent"}"#,
            r#"{"kind":"post_label","label":"nsfahighprecision","lifetime":"permanent"}"#,
            r#"{"kind":"post_label","label":"nsfwhighprecision","lifetime":"permanent"}"#,
            r#"{"kind":"post_label","label":"NsfwHighPrecision"}"#,
            r#"{"kind":"post_label","label":"NsfwHighPrecision","lifetime":"90d"}"#,
            r#"{"kind":"post_label","lifetime":"permanent"}"#,
            r#"{"kind":"post_label","label":"NsfwHighPrecision","lifetime":"7d","ttl":1}"#,
        ] {
            assert!(serde_json::from_str::<PageAction>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn every_listed_label_is_on_the_post_menu_with_every_lifetime() {
        let post = PageAction::menu(ControlEntity::Post);
        let user = PageAction::menu(ControlEntity::User);
        for label in PostLabelName::all() {
            for lifetime in [Lifetime::Days7, Lifetime::Days30, Lifetime::Permanent] {
                let a = PageAction::PostLabel { label, lifetime };
                assert!(post.contains(&a), "{a:?}");
                assert!(!user.contains(&a), "{a:?}");
            }
        }
        assert_eq!(
            post.iter().filter(|a| a.is_post_label()).count(),
            3 * PAGE_POST_LABELS.len()
        );
    }

    #[test]
    fn listed_labels_never_shadow_an_action_of_their_own() {
        let mut seen = std::collections::HashSet::new();
        for label in PAGE_POST_LABELS {
            assert!(seen.insert(label), "{label} listed twice");
            assert!(
                !["SpamHighRecall", "RiskyHighVizReply"].contains(label),
                "{label} has its own page action"
            );
            let kinds: Vec<&str> = PageAction::menu(ControlEntity::Post)
                .into_iter()
                .chain(PageAction::menu(ControlEntity::User))
                .filter(|a| !a.is_post_label())
                .map(PageAction::kind)
                .collect();
            assert!(!kinds.contains(label), "{label} is another action's kind");
        }
    }

    #[test]
    fn a_record_with_a_post_label_is_refused_by_older_images() {
        let json = r#"{"v":3,"entity_type":"post","head":"H","mode":"live","actions":[{"kind":"hide_reply","lifetime":"permanent"}],"version":1,"confidence":"high"}"#;
        let mut r = ActionControlRecord::parse_strict(json.as_bytes()).unwrap();
        assert!(!r.uses_post_label());
        assert_eq!(r.schema_version(), ROWS_RECORD_VERSION);
        r.actions = vec![PageAction::PostLabel {
            label: PostLabelName::parse("NsfwHighPrecision").unwrap(),
            lifetime: Lifetime::Permanent,
        }];
        r.rollout_bp = Some(1_000);
        assert_eq!(r.schema_version(), 7);
        assert_eq!(r.schema_version(), POST_LABEL_RECORD_VERSION);
        assert!(r.schema_version() > SUSPENSION_OPTIONS_RECORD_VERSION);
        r.v = r.schema_version();
        let back = ActionControlRecord::parse_strict(serde_json::to_string(&r).unwrap().as_bytes())
            .unwrap();
        assert_eq!(back, r);
    }
}
