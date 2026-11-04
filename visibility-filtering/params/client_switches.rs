use super::limited_actions_policy::REASON_FIELD;
use super::LimitedActionsPolicies;
use crate::metrics::Metrics;
use crate::models::{ClientCapability, ClientSwitch, LimitedEngagementReason, VerifyBlurSupport};
use crate::params::limited_actions_copy::LimitedActionsCopy;
use arc_swap::ArcSwap;
use std::sync::Arc;
use strum::VariantArray;
use xai_feature_switches::{FeatureSwitches, RecipientBuilder, Version};
use xai_twittercontext_proto::TwitterContextViewer;

const SOFT_NSFW_NEW_USER_DROP: &str = "soft_nsfw_tweet_label_new_user_rule_enabled";

const fn key(switch: ClientSwitch) -> &'static str {
    match switch {
        ClientSwitch::ModernBlur => {
            "media_visibility_treatments_blurred_media_interstitial_enabled"
        }
        ClientSwitch::StaleTweetLimits => "stale_tweet_limited_actions_rules_enabled",
        ClientSwitch::CommunityViewerRemovedLimits => {
            "community_tweet_viewer_removed_limited_actions_rules_enabled"
        }
        ClientSwitch::GoreBlurIgnoresSettings => {
            "media_visibility_treatments_blurred_media_interstitial_ignore_settings_enabled"
        }
        ClientSwitch::FosnrRules => "freedom_of_speech_not_reach_rules_enabled",
        ClientSwitch::FosnrFallbackDrops => {
            "freedom_of_speech_not_reach_fallback_drop_rules_enabled"
        }
    }
}

const fn age_verification_key(support: VerifyBlurSupport) -> &'static str {
    match support {
        VerifyBlurSupport::Unsupported => "age_verification_uk_tombstone_rule_enabled",
        VerifyBlurSupport::IosNeedsUpdate => "age_verification_ios_tombstone_rule_enabled",
        VerifyBlurSupport::AndroidNeedsUpdate => "age_verification_android_tombstone_rule_enabled",
        VerifyBlurSupport::Supported => "age_verification_blurred_media_interstitial_rule_enabled",
    }
}

const fn age_verification_default_on(support: VerifyBlurSupport) -> bool {
    match support {
        VerifyBlurSupport::Unsupported => true,
        VerifyBlurSupport::IosNeedsUpdate
        | VerifyBlurSupport::AndroidNeedsUpdate
        | VerifyBlurSupport::Supported => false,
    }
}

#[derive(Clone)]
pub(crate) struct ClientSwitches {
    feature_switches: Arc<ArcSwap<FeatureSwitches>>,
    copy: Arc<LimitedActionsCopy>,
}

impl ClientSwitches {
    pub fn new(feature_switches: Arc<ArcSwap<FeatureSwitches>>, copy: LimitedActionsCopy) -> Self {
        Self {
            feature_switches,
            copy: Arc::new(copy),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Self::for_tests_with(LimitedActionsCopy::from_json(
            crate::params::limited_actions_copy::tests::BUNDLE,
        ))
    }

    #[cfg(test)]
    pub(crate) fn for_tests_with(copy: LimitedActionsCopy) -> Self {
        const SCALA_RULES_COPY: &str =
            include_str!("../../tests/fixtures/scala_client_switches.yml");
        const SCALA_POLICY_COPY: &str =
            include_str!("../../tests/fixtures/limited_actions_policy.yml");
        let features = [SCALA_RULES_COPY, SCALA_POLICY_COPY]
            .iter()
            .flat_map(|yaml| xai_feature_switches::load_yaml_string(yaml).unwrap())
            .collect();
        Self::new(
            Arc::new(ArcSwap::from_pointee(
                FeatureSwitches::new(features).unwrap(),
            )),
            copy,
        )
    }

    pub fn resolve(
        &self,
        context: Option<&TwitterContextViewer>,
        viewer_id: Option<u64>,
        country_code: Option<&str>,
    ) -> ClientCapability {
        if context.is_none() {
            return ClientCapability::default();
        }
        let results = self
            .feature_switches
            .load()
            .match_recipient(&recipient(context, viewer_id, country_code).build());
        let on =
            |key: &str, default_on: bool| results.get_bool_no_impression(key).unwrap_or(default_on);
        ClientCapability {
            verify_blur_support: verify_blur_support(|support| {
                on(
                    age_verification_key(support),
                    age_verification_default_on(support),
                )
            }),
            switches_on: ClientSwitch::VARIANTS
                .iter()
                .copied()
                .filter(|&switch| on(key(switch), switch.default_on()))
                .collect(),
        }
    }

    pub fn soft_nsfw_new_user_drop(
        &self,
        viewer_id: Option<u64>,
        country_code: Option<&str>,
    ) -> bool {
        let Some(viewer_id) = viewer_id else {
            return false;
        };
        self.feature_switches
            .load()
            .match_recipient(&recipient(None, Some(viewer_id), country_code).build())
            .get_bool(SOFT_NSFW_NEW_USER_DROP)
            .unwrap_or(false)
    }

    pub fn limited_actions_policies(
        &self,
        context: Option<&TwitterContextViewer>,
        viewer_id: Option<u64>,
        country_code: Option<&str>,
        reasons: impl IntoIterator<Item = LimitedEngagementReason>,
        metrics: &Metrics,
    ) -> LimitedActionsPolicies {
        let feature_switches = self.feature_switches.load();
        let language = context
            .map(|context| context.request_language_code.as_str())
            .filter(|language| !language.is_empty());
        LimitedActionsPolicies::resolve(
            reasons,
            |reason| {
                feature_switches.match_recipient(
                    &recipient(context, viewer_id, country_code)
                        .opt_language(language)
                        .custom_string(REASON_FIELD, reason.limited_actions_string())
                        .build(),
                )
            },
            |action_type, prompt_copy| {
                self.copy
                    .prompt(action_type, prompt_copy, language, country_code)
            },
            metrics,
        )
        .in_language(language)
    }
}

fn recipient(
    context: Option<&TwitterContextViewer>,
    viewer_id: Option<u64>,
    country_code: Option<&str>,
) -> RecipientBuilder {
    let mut recipient = RecipientBuilder::new().opt_country(country_code);
    if let Some(viewer_id) = viewer_id {
        recipient = recipient.user_id(viewer_id);
    }
    if let Some(context) = context {
        recipient = recipient
            .client_version_opt(client_version(&context.user_agent).map(|v| v.to_string()));
        if context.client_application_id != 0 {
            recipient = recipient.client_app_id(context.client_application_id);
        }
    }
    recipient
}

fn verify_blur_support(on: impl Fn(VerifyBlurSupport) -> bool) -> Option<VerifyBlurSupport> {
    VerifyBlurSupport::VARIANTS
        .iter()
        .copied()
        .find(|&support| on(support))
}

fn client_version(user_agent: &str) -> Option<Version> {
    let (client_name, rest) = user_agent.split_once('/')?;
    if client_name == "Mozilla" {
        return None;
    }
    Version::parse(rest.split([' ', '(', ')']).next()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rules::fixtures::CLIENT_CLASSES;

    #[test]
    fn a_viewer_in_the_ddg_treatment_gets_the_switch_and_one_impression() {
        use crate::params::{Experiments, SwitchFiles};
        use xai_feature_switches::{ReloadableAbDeciderStore, SpyingBucketImpressor};
        let root = tempfile::tempdir().unwrap();
        let abdecider = root.path().join("abdecider.yml");
        let all_slots = "f".repeat(1280);
        std::fs::write(
            &abdecider,
            format!(
                "_common: &common\n  soft_nsfw_ddg_1:\n    version: 1\n    logged_in_only: true\n    darkmode: false\n    group: g\n    group_bucket_strategy: splitted\n    diversion: user_id\n    experiment_id: 1\n    buckets:\n    - name: treatment\n      value: 10000\n      allocated_slots: \"{all_slots}\"\nproduction:\n  <<: *common\n"
            ),
        )
        .unwrap();
        let fs_path = root.path().join("rust_vf.yml");
        std::fs::write(
            root.path().join("soft_nsfw.yml"),
            "soft_nsfw:\n  parameters:\n    soft_nsfw_tweet_label_new_user_rule_enabled: {type: boolean, default: false}\n  rules:\n  - query: \"[soft_nsfw_ddg_1 bucket_membership treatment]\"\n    values: {soft_nsfw_tweet_label_new_user_rule_enabled: true}\n",
        )
        .unwrap();
        let impressor = Arc::new(SpyingBucketImpressor::new());
        let experiments = Experiments::from_parts(
            ReloadableAbDeciderStore::new(&abdecider).unwrap(),
            Arc::clone(&impressor) as _,
        );
        let engine = SwitchFiles::beside(fs_path.to_str().unwrap(), experiments)
            .load(&Metrics::none())
            .unwrap();
        let switches = ClientSwitches::new(
            Arc::new(ArcSwap::from_pointee(engine)),
            LimitedActionsCopy::from_json("[]"),
        );

        assert!(switches.soft_nsfw_new_user_drop(Some(TEST_USER_ID), Some("us")));
        assert_eq!(impressor.impression_count("soft_nsfw_ddg_1"), 1);
        assert!(!switches.soft_nsfw_new_user_drop(None, Some("us")));
        assert_eq!(impressor.impression_count("soft_nsfw_ddg_1"), 1);
    }

    const TEST_USER_ID: u64 = 1_000_000_000_000_000_001;

    fn context(app_id: i64, user_agent: &str) -> TwitterContextViewer {
        TwitterContextViewer {
            client_application_id: app_id,
            user_agent: user_agent.into(),
            ..TwitterContextViewer::default()
        }
    }

    #[test]
    fn client_version_is_major_minor_patch_of_an_app_user_agent() {
        let version = |user_agent: &str| client_version(user_agent).map(|v| v.to_string());
        let ios = |version: &str| format!("Twitter-iPhone/{version} iOS/17.0 (Apple;iPhone15,2)");
        assert_eq!(version(&ios("11.11.5")).as_deref(), Some("11.11.5"));
        assert_eq!(version(&ios("11.11.5.2")).as_deref(), Some("11.11.5"));
        assert_eq!(version(&ios("6.50-Enterprise")).as_deref(), Some("6.50"));
        assert_eq!(
            version("TwitterAndroid/11.11.0-release.00 (311110000-r-0) Pixel 7/14 (Google)")
                .as_deref(),
            Some("11.11.0")
        );
        assert_eq!(
            version("Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0 Safari/537.36"),
            None
        );
        assert_eq!(version("curl"), None);
        assert_eq!(version(""), None);
    }

    #[test]
    fn each_client_class_resolves_from_its_app_id_and_user_agent() {
        let switches = ClientSwitches::for_tests();
        for class in CLIENT_CLASSES {
            let client = context(class.app_id, class.user_agent);
            assert_eq!(
                switches.resolve(Some(&client), Some(1), Some("fr")),
                class.capability,
                "{}",
                class.name
            );
        }

        let third_party = context(0, "ThirdPartyClient/2.0");
        assert_eq!(
            switches.resolve(Some(&third_party), Some(TEST_USER_ID), None),
            ClientCapability {
                verify_blur_support: Some(VerifyBlurSupport::Supported),
                ..ClientCapability::DEFAULT.with(ClientSwitch::ModernBlur)
            }
        );
        let app_id = |name: &str| {
            CLIENT_CLASSES
                .iter()
                .find(|class| class.name == name)
                .unwrap()
                .app_id
        };
        let ios = app_id("ios_current");
        for (version, stale_tweet_limits) in [("9.19.9", false), ("9.20.0", true)] {
            let client = context(ios, &format!("Twitter-iPhone/{version} iOS/17.0"));
            assert_eq!(
                switches
                    .resolve(Some(&client), Some(1), None)
                    .has(ClientSwitch::StaleTweetLimits),
                stale_tweet_limits,
                "{version}"
            );
        }
        let android = app_id("android_current");
        for (user_agent, community_viewer_removed_limits) in [
            ("Twitter-iPhone/10.15.9 iOS/17.0", false),
            ("Twitter-iPhone/10.16.0 iOS/17.0", true),
            ("TwitterAndroid/10.15.9-release.00 (310159000-r-0)", false),
            ("TwitterAndroid/10.16.0-release.00 (310160000-r-0)", true),
        ] {
            let app_id = if user_agent.starts_with("TwitterAndroid") {
                android
            } else {
                ios
            };
            assert_eq!(
                switches
                    .resolve(Some(&context(app_id, user_agent)), Some(1), None)
                    .has(ClientSwitch::CommunityViewerRemovedLimits),
                community_viewer_removed_limits,
                "{user_agent}"
            );
        }
        let web = &CLIENT_CLASSES[0];
        assert!(!switches
            .resolve(Some(&context(web.app_id, web.user_agent)), None, None)
            .has(ClientSwitch::CommunityViewerRemovedLimits));
        assert_eq!(
            switches.resolve(None, Some(TEST_USER_ID), None),
            ClientCapability::default()
        );
        let no_keys = ClientSwitches::new(
            Arc::new(ArcSwap::from_pointee(FeatureSwitches::null().unwrap())),
            LimitedActionsCopy::from_json("[]"),
        );
        let web = &CLIENT_CLASSES[0];
        assert_eq!(
            no_keys.resolve(Some(&context(web.app_id, web.user_agent)), None, None),
            ClientCapability::default()
        );
    }

    #[test]
    fn the_most_severe_switch_on_wins() {
        use VerifyBlurSupport::{AndroidNeedsUpdate, IosNeedsUpdate, Supported, Unsupported};
        for (on, expected) in [
            (&[Supported][..], Some(Supported)),
            (&[Supported, IosNeedsUpdate], Some(IosNeedsUpdate)),
            (&[AndroidNeedsUpdate, Supported], Some(AndroidNeedsUpdate)),
            (&[AndroidNeedsUpdate, Unsupported], Some(Unsupported)),
            (&[], None),
        ] {
            assert_eq!(
                verify_blur_support(|support| on.contains(&support)),
                expected,
                "{on:?}"
            );
        }
    }
}
