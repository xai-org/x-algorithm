mod client_switches;
pub(crate) mod limited_actions_copy;
pub(crate) mod limited_actions_policy;

use crate::metrics::families::FEATURE_SWITCH_LOAD_FAILURES;
use crate::metrics::Metrics;
use arc_swap::ArcSwap;
use limited_actions_copy::DriftCheck;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use strum::VariantArray;
use xai_feature_switches::{
    parse_query, DdgExperimentBucketsChooser, ExperimentBucketImpressor, ExperimentBucketsChooser,
    Feature, FeatureSwitches, KafkaBucketImpressor, NullBucketImpressor,
    NullExperimentBucketsChooser, QueryNode, RecipientBuilder, ReloadableAbDeciderStore, Value,
    DEFAULT_ABDECIDER_PATH,
};
use xai_safety_label_source::{LabelCacheSwitches, SafetyLabelSource};

pub(crate) use client_switches::ClientSwitches;
pub(crate) use limited_actions_policy::LimitedActionsPolicies;

const SCALA_FILES: [&str; 7] = [
    "age_verification.yml",
    "community_tweet.yml",
    "country_specific_nsfw_content_gating.yml",
    "freedom_of_speech_not_reach.yml",
    "media_visibility_treatments.yml",
    "soft_nsfw.yml",
    "stale_tweet.yml",
];

const LIMITED_ACTIONS_POLICY_FILE: &str =
    "../../visibility-limited-actions/main/limited_actions_policy.yml";

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum CountryList {
    NsfwGating,
    AgeVerification,
    Tombstone,
    LocalRegulations,
}

impl CountryList {
    const fn key(self) -> &'static str {
        match self {
            Self::NsfwGating => "country_specific_nsfw_content_gating_countries",
            Self::AgeVerification => {
                "country_specific_nsfw_content_gating_age_verification_countries"
            }
            Self::Tombstone => "country_specific_nsfw_content_gating_tombstone_countries",
            Self::LocalRegulations => {
                "country_specific_nsfw_content_gating_local_regulations_countries"
            }
        }
    }

    pub(crate) fn names_by_default(self, code: &str) -> Option<&'static str> {
        self.default_codes()
            .iter()
            .find(|&&listed| listed == code)
            .copied()
    }

    pub(crate) fn listed_by_default(code: &str) -> Option<&'static str> {
        Self::VARIANTS
            .iter()
            .find_map(|list| list.names_by_default(code))
    }

    const fn default_codes(self) -> &'static [&'static str] {
        const AGE_VERIFICATION: &[&str] = &[
            "at", "au", "be", "bg", "br", "cy", "cz", "de", "dk", "ee", "es", "fi", "fr", "gb",
            "gr", "hr", "hu", "ie", "it", "lt", "lu", "lv", "mt", "nl", "pl", "pt", "ro", "se",
            "si", "sk",
        ];
        match self {
            Self::NsfwGating => &[
                "ar", "au", "br", "ca", "de", "es", "fr", "gb", "id", "it", "kr", "mx", "nl", "ph",
                "pt", "th",
            ],
            Self::AgeVerification | Self::Tombstone => AGE_VERIFICATION,
            Self::LocalRegulations => &["in"],
        }
    }
}

pub(crate) struct CountryLists {
    lists: ArcSwap<Vec<Vec<String>>>,
}

impl CountryLists {
    pub fn starting_at_default() -> Self {
        let lists = CountryList::VARIANTS
            .iter()
            .map(|list| owned(list.default_codes()))
            .collect();
        Self {
            lists: ArcSwap::from_pointee(lists),
        }
    }

    pub fn contains(&self, list: CountryList, country_code: &str) -> bool {
        #[expect(
            clippy::indexing_slicing,
            reason = "`lists` maps `CountryList::VARIANTS`, which is in declaration order"
        )]
        let codes = &self.lists.load()[list as usize];
        codes.iter().any(|c| c == country_code)
    }

    pub fn refresh(&self, feature_switches: &FeatureSwitches) {
        let results = feature_switches.match_recipient(&RecipientBuilder::new().build());
        let lists = CountryList::VARIANTS
            .iter()
            .map(|&list| load(list, results.get_array_no_impression(list.key())))
            .collect();
        self.lists.store(Arc::new(lists));
    }
}

pub(crate) fn label_cache_switches(feature_switches: &FeatureSwitches) -> LabelCacheSwitches {
    let results = feature_switches.match_recipient(&RecipientBuilder::new().build());
    let value = |key| {
        results
            .get_i64_no_impression(key)
            .and_then(|value| u64::try_from(value).ok())
    };
    let default = LabelCacheSwitches::default();
    LabelCacheSwitches {
        shadow_buckets: value("rust_vf_safety_label_cache_shadow_buckets")
            .unwrap_or(default.shadow_buckets),
        serve_buckets: value("rust_vf_safety_label_cache_serve_buckets")
            .unwrap_or(default.serve_buckets),
        found_ttl: value("rust_vf_safety_label_cache_found_ttl_secs")
            .map_or(default.found_ttl, Duration::from_secs),
        not_found_ttl: value("rust_vf_safety_label_cache_not_found_ttl_secs")
            .map_or(default.not_found_ttl, Duration::from_secs),
        refresh_cutoff_ms: value("rust_vf_safety_label_cache_refresh_cutoff_ms")
            .unwrap_or(default.refresh_cutoff_ms),
        replay_bucket: value("rust_vf_safety_label_cache_replay_bucket"),
        shadow_mh_check_keys_per_sec: value(
            "rust_vf_safety_label_cache_shadow_mh_check_keys_per_sec",
        )
        .map_or(default.shadow_mh_check_keys_per_sec, |keys| {
            u32::try_from(keys).unwrap_or(u32::MAX)
        }),
        shadow_fill_keys_per_sec: value("rust_vf_safety_label_cache_shadow_fill_keys_per_sec")
            .map_or(default.shadow_fill_keys_per_sec, |keys| {
                u32::try_from(keys).unwrap_or(u32::MAX)
            }),
    }
}

#[derive(Clone)]
pub(crate) struct Experiments {
    store: Option<Arc<ReloadableAbDeciderStore>>,
    impressor: Arc<dyn ExperimentBucketImpressor>,
}

impl Experiments {
    pub fn none() -> Self {
        Self {
            store: None,
            impressor: Arc::new(NullBucketImpressor::new()),
        }
    }

    pub async fn serving(datacenter: &str) -> Self {
        Self {
            store: abdecider_store(),
            impressor: kafka_impressor(datacenter).await,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_parts(
        store: Arc<ReloadableAbDeciderStore>,
        impressor: Arc<dyn ExperimentBucketImpressor>,
    ) -> Self {
        Self {
            store: Some(store),
            impressor,
        }
    }

    fn engine(&self, features: Vec<Feature>) -> xai_feature_switches::Result<FeatureSwitches> {
        let chooser: Arc<dyn ExperimentBucketsChooser> = match &self.store {
            Some(store) => Arc::new(
                DdgExperimentBucketsChooser::new(Arc::clone(store) as _)
                    .with_key_filter(experiment_keys(&features)),
            ),
            None => Arc::new(NullExperimentBucketsChooser::new()),
        };
        FeatureSwitches::with_options(features, chooser, Arc::clone(&self.impressor), None, false)
    }
}

fn abdecider_store() -> Option<Arc<ReloadableAbDeciderStore>> {
    match ReloadableAbDeciderStore::new(DEFAULT_ABDECIDER_PATH) {
        Ok(store) => {
            store.start_reload_task();
            Some(store)
        }
        Err(error) => {
            tracing::warn!(%error, "feature switches: no abdecider store; bucket rules never match");
            None
        }
    }
}

async fn kafka_impressor(datacenter: &str) -> Arc<dyn ExperimentBucketImpressor> {
    let producer = FeatureSwitches::builder()
        .without_abdecider()
        .datacenter(datacenter)
        .kafka_impressor_enabled(true)
        .build()
        .await
        .ok()
        .and_then(|engine| engine.kafka_producer().cloned());
    match producer {
        Some(producer) => Arc::new(KafkaBucketImpressor::new(producer)),
        None => {
            tracing::warn!("feature switches: no Kafka producer; DDG impressions are dropped");
            Arc::new(NullBucketImpressor::new())
        }
    }
}

fn experiment_keys(features: &[Feature]) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for feature in features {
        for rule in &feature.rules {
            let query = rule.query.as_deref().unwrap_or_default();
            if let Ok(node) = parse_query(query, &feature.name) {
                collect_keys(&node, &mut keys);
            }
        }
    }
    keys
}

fn collect_keys(node: &QueryNode, keys: &mut BTreeSet<String>) {
    match node {
        QueryNode::Criteria(criteria) => {
            let query = criteria.to_query();
            let mut parts = query.trim_matches(['[', ']']).split_whitespace();
            if let (
                Some(key),
                Some(
                    "bucket_membership"
                    | "bucket_membership_no_impression"
                    | "experiment_membership",
                ),
            ) = (parts.next(), parts.next())
            {
                keys.insert(key.to_string());
            }
        }
        QueryNode::And(children) | QueryNode::Or(children) => {
            children.iter().for_each(|child| collect_keys(child, keys));
        }
        QueryNode::Not(child) => collect_keys(child, keys),
        QueryNode::AlwaysTrue | QueryNode::Never => {}
    }
}

pub(crate) struct SwitchFiles {
    files: Vec<(PathBuf, Vec<Feature>)>,
    experiments: Experiments,
}

impl SwitchFiles {
    pub fn beside(fs_path: &str, experiments: Experiments) -> Self {
        let fs_path = Path::new(fs_path);
        let dir = fs_path.parent().unwrap_or_else(|| Path::new(""));
        let files = std::iter::once(fs_path.to_path_buf())
            .chain(SCALA_FILES.iter().map(|name| dir.join(name)))
            .chain(std::iter::once(dir.join(LIMITED_ACTIONS_POLICY_FILE)))
            .map(|path| (path, Vec::new()))
            .collect();
        Self { files, experiments }
    }

    pub(crate) fn load(&mut self, metrics: &Metrics) -> Option<FeatureSwitches> {
        for (path, last_good) in &mut self.files {
            match parse_file(path) {
                Ok(features) => *last_good = features,
                Err(error) => {
                    let file = path.file_name().and_then(|name| name.to_str());
                    load_failed(file.unwrap_or_default(), &error, metrics);
                }
            }
        }
        let features = self
            .files
            .iter()
            .flat_map(|(_, features)| features.iter().cloned())
            .collect();
        self.experiments
            .engine(features)
            .map_err(|error| load_failed("all", &error, metrics))
            .ok()
    }
}

fn parse_file(path: &Path) -> xai_feature_switches::Result<Vec<Feature>> {
    let features = xai_feature_switches::load_yaml_file(path)?;
    FeatureSwitches::new(features.clone())?;
    Ok(features)
}

fn load_failed(file: &str, error: &xai_feature_switches::Error, metrics: &Metrics) {
    tracing::warn!(
        file,
        %error,
        "feature switches: file failed to load; keeping its last good values"
    );
    metrics.count(&FEATURE_SWITCH_LOAD_FAILURES, [file], 1);
}

pub(crate) fn spawn_refresh(
    mut files: SwitchFiles,
    feature_switches: Arc<ArcSwap<FeatureSwitches>>,
    country_lists: Arc<CountryLists>,
    safety_label_source: Arc<SafetyLabelSource>,
    copy_drift: DriftCheck,
    metrics: Metrics,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            if let Some(engine) = files.load(&metrics) {
                feature_switches.store(Arc::new(engine));
            }
            country_lists.refresh(&feature_switches.load());
            safety_label_source
                .set_label_cache_switches(label_cache_switches(&feature_switches.load()));
            copy_drift.run(&metrics);
        }
    });
}

fn owned(codes: &[&str]) -> Vec<String> {
    codes.iter().map(|code| code.to_string()).collect()
}

fn parse(values: &[Value]) -> Option<Vec<String>> {
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|code| code.len() == 2 && code.bytes().all(|b| b.is_ascii_alphabetic()))
                .map(str::to_ascii_lowercase)
        })
        .collect()
}

fn load(list: CountryList, configured: Option<&Vec<Value>>) -> Vec<String> {
    let Some(values) = configured else {
        return owned(list.default_codes());
    };
    parse(values).unwrap_or_else(|| {
        tracing::warn!(
            key = list.key(),
            configured = ?values,
            "country list: an entry is not a two-letter code; using the default"
        );
        owned(list.default_codes())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(yaml: &str) -> FeatureSwitches {
        FeatureSwitches::load_string(yaml).unwrap()
    }

    #[test]
    fn refresh_reads_each_key_and_falls_back_to_its_default() {
        let lists = CountryLists::starting_at_default();
        assert!(lists.contains(CountryList::NsfwGating, "de"));
        assert!(lists.contains(CountryList::LocalRegulations, "in"));
        assert!(!lists.contains(CountryList::NsfwGating, "xx"));

        lists.refresh(&engine(
            r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_countries:
      type: array
      default:
      - "XX"
    country_specific_nsfw_content_gating_local_regulations_countries:
      type: array
      default: []
    country_specific_nsfw_content_gating_tombstone_countries:
      type: string
      default: "xx"
"#,
        ));
        assert!(lists.contains(CountryList::NsfwGating, "xx"));
        assert!(!lists.contains(CountryList::NsfwGating, "de"));
        assert!(lists.contains(CountryList::AgeVerification, "fr"));
        assert!(!lists.contains(CountryList::LocalRegulations, "in"));
        assert!(lists.contains(CountryList::Tombstone, "fr"));
        assert!(!lists.contains(CountryList::Tombstone, "xx"));

        lists.refresh(&engine("other:\n  parameters: {}\n"));
        assert!(lists.contains(CountryList::NsfwGating, "de"));
        assert!(!lists.contains(CountryList::NsfwGating, "xx"));
        assert!(lists.contains(CountryList::LocalRegulations, "in"));
    }

    #[test]
    fn label_cache_switches_read_rust_vf_and_keep_defaults_for_missing_or_negative_values() {
        assert_eq!(
            label_cache_switches(&engine("rust_vf:\n  parameters: {}\n")),
            LabelCacheSwitches::default()
        );
        assert_eq!(
            label_cache_switches(&engine(
                r#"
rust_vf:
  parameters:
    rust_vf_safety_label_cache_shadow_buckets: {type: integer, default: 100}
    rust_vf_safety_label_cache_serve_buckets: {type: integer, default: 10}
    rust_vf_safety_label_cache_found_ttl_secs: {type: integer, default: 3600}
    rust_vf_safety_label_cache_not_found_ttl_secs: {type: integer, default: -5}
    rust_vf_safety_label_cache_refresh_cutoff_ms: {type: long, default: 1700000000000}
    rust_vf_safety_label_cache_replay_bucket: {type: integer, default: 7}
    rust_vf_safety_label_cache_shadow_mh_check_keys_per_sec: {type: integer, default: 50}
    rust_vf_safety_label_cache_shadow_fill_keys_per_sec: {type: integer, default: 200}
"#
            )),
            LabelCacheSwitches {
                shadow_buckets: 100,
                serve_buckets: 10,
                found_ttl: Duration::from_secs(3600),
                refresh_cutoff_ms: 1_700_000_000_000,
                replay_bucket: Some(7),
                shadow_mh_check_keys_per_sec: 50,
                shadow_fill_keys_per_sec: 200,
                ..LabelCacheSwitches::default()
            }
        );
    }

    #[test]
    fn one_invalid_entry_falls_back_to_the_default() {
        let lists = CountryLists::starting_at_default();
        lists.refresh(&engine(
            r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_age_verification_countries:
      type: array
      default: ["xx", " fr"]
    country_specific_nsfw_content_gating_tombstone_countries:
      type: array
      default: ["xx", "fra"]
    country_specific_nsfw_content_gating_local_regulations_countries:
      type: array
      default: ["xx", 7]
"#,
        ));
        for (list, default) in [
            (CountryList::AgeVerification, "fr"),
            (CountryList::Tombstone, "fr"),
            (CountryList::LocalRegulations, "in"),
        ] {
            assert!(lists.contains(list, default), "{list:?}");
            assert!(!lists.contains(list, "xx"), "{list:?}");
        }
    }

    #[test]
    fn a_file_that_fails_to_load_keeps_its_last_good_values() {
        const AGE_VERIFICATION: &str = r#"
age_verification:
  parameters:
    age_verification_ios_tombstone_rule_enabled: {type: boolean, default: true}
"#;
        const COUNTRIES: &str = r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_local_regulations_countries:
      type: array
      default: ["us"]
"#;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("visibility/main");
        let policy_dir = root.path().join("visibility-limited-actions/main");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&policy_dir).unwrap();
        let write = |name: &str, yaml: &str| std::fs::write(dir.join(name), yaml).unwrap();
        write("age_verification.yml", AGE_VERIFICATION);
        write("country_specific_nsfw_content_gating.yml", COUNTRIES);
        write(
            "other.yml",
            "other:\n  parameters:\n    other_flag: {type: boolean, default: true}\n",
        );
        let fs_path = dir.join("rust_vf.yml");
        let mut files = SwitchFiles::beside(fs_path.to_str().unwrap(), Experiments::none());
        let (metrics, recording) = Metrics::recording();
        let ios_tombstone = |engine: &FeatureSwitches| {
            engine
                .match_recipient(&RecipientBuilder::new().build())
                .get_bool_no_impression("age_verification_ios_tombstone_rule_enabled")
        };
        let failed = |file: &str| format!("feature_switch_load_failures{{file={file}}} 1");
        let failures = || -> Vec<String> {
            recording
                .take(&FEATURE_SWITCH_LOAD_FAILURES.family)
                .iter()
                .map(ToString::to_string)
                .collect()
        };

        let engine = files.load(&metrics).unwrap();
        assert_eq!(ios_tombstone(&engine), Some(true));
        assert!(!engine
            .match_recipient(&RecipientBuilder::new().build())
            .is_set("other_flag"));
        assert_eq!(
            failures(),
            [
                failed("rust_vf.yml"),
                failed("community_tweet.yml"),
                failed("freedom_of_speech_not_reach.yml"),
                failed("media_visibility_treatments.yml"),
                failed("soft_nsfw.yml"),
                failed("stale_tweet.yml"),
                failed("limited_actions_policy.yml"),
            ]
        );

        write("rust_vf.yml", "rust_vf:\n  parameters: {}\n");
        write(
            "media_visibility_treatments.yml",
            "media_visibility_treatments:\n  parameters: {}\n",
        );
        write("stale_tweet.yml", "stale_tweet:\n  parameters: {}\n");
        write("soft_nsfw.yml", "soft_nsfw:\n  parameters: {}\n");
        write(
            "community_tweet.yml",
            "community_tweet:\n  parameters: {}\n",
        );
        write(
            "freedom_of_speech_not_reach.yml",
            "freedom_of_speech_not_reach:\n  parameters: {}\n",
        );
        std::fs::write(
            policy_dir.join("limited_actions_policy.yml"),
            "limited_actions_policy:\n  parameters: {}\n",
        )
        .unwrap();
        for (broken, code) in [
            (
                "age_verification:\n  parameters: {}\n  rules:\n  - query: \"([user_id in 1]\"\n    values: {}\n",
                "xx",
            ),
            ("age_verification: [", "yy"),
        ] {
            write("age_verification.yml", broken);
            write(
                "country_specific_nsfw_content_gating.yml",
                &COUNTRIES.replace("us", code),
            );
            let engine = files.load(&metrics).unwrap();
            assert_eq!(ios_tombstone(&engine), Some(true), "{broken}");
            let lists = CountryLists::starting_at_default();
            lists.refresh(&engine);
            assert!(
                lists.contains(CountryList::LocalRegulations, code),
                "{broken}"
            );
            assert_eq!(failures(), [failed("age_verification.yml")], "{broken}");
        }
    }
}
