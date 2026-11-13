use super::LimitedActionsPolicies;
use crate::metrics::Metrics;
use crate::models::{
    ClientCapability, ClientSwitch, LimitedEngagementReason, RequestFacts, RequestViewer,
    VerifyBlurSupport,
};
use crate::params::limited_actions_copy::LimitedActionsCopy;
use crate::params::SwitchEngines;
use crate::rules::{ExperimentSet, RequestExperiments};
use arc_swap::ArcSwap;
use std::sync::Arc;
use strum::VariantArray;
use xai_feature_switches::{flavors_for_app_id, ClientFlavors, FeatureSwitchResults, Recipient};

const SOFT_NSFW_NEW_USER_DROP: &str = "soft_nsfw_tweet_label_new_user_rule_enabled";
const REASON_FIELD: &str = "LimitedEngagementReason";

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

#[derive(Clone)]
pub(crate) struct ClientSwitches {
    feature_switches: Arc<ArcSwap<SwitchEngines>>,
    copy: Arc<LimitedActionsCopy>,
}

impl ClientSwitches {
    pub fn new(feature_switches: Arc<ArcSwap<SwitchEngines>>, copy: LimitedActionsCopy) -> Self {
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
    pub(crate) fn in_ddg_arm(
        ddg: &str,
        arm: &str,
        files: &[(&str, &str)],
        experiments: impl FnOnce(
            Arc<xai_feature_switches::ReloadableAbDeciderStore>,
        ) -> crate::params::Experiments,
    ) -> Self {
        use crate::params::SwitchFiles;
        let root = tempfile::tempdir().unwrap();
        let abdecider = root.path().join("abdecider.yml");
        let all_slots = "f".repeat(1280);
        std::fs::write(
            &abdecider,
            format!(
                "_common: &common\n  {ddg}:\n    version: 1\n    logged_in_only: true\n    darkmode: false\n    group: g\n    group_bucket_strategy: splitted\n    diversion: user_id\n    experiment_id: 1\n    buckets:\n    - name: {arm}\n      value: 10000\n      allocated_slots: \"{all_slots}\"\nproduction:\n  <<: *common\n"
            ),
        )
        .unwrap();
        for (name, yaml) in files {
            std::fs::write(root.path().join(name), yaml).unwrap();
        }
        let experiments =
            experiments(xai_feature_switches::ReloadableAbDeciderStore::new(&abdecider).unwrap());
        let fs_path = root.path().join("rust_vf.yml");
        let engines = SwitchFiles::beside(fs_path.to_str().unwrap(), experiments)
            .load(&Metrics::none())
            .unwrap();
        Self::new(
            Arc::new(ArcSwap::from_pointee(engines)),
            LimitedActionsCopy::from_json("[]"),
        )
    }

    #[cfg(test)]
    pub(crate) fn for_tests_with(copy: LimitedActionsCopy) -> Self {
        use xai_feature_switches::FeatureSwitches;
        const SCALA_RULES_COPY: &str =
            include_str!("../../tests/fixtures/scala_client_switches.yml");
        const SCALA_POLICY_COPY: &str =
            include_str!("../../tests/fixtures/limited_actions_policy.yml");
        Self::new(
            Arc::new(ArcSwap::from_pointee(SwitchEngines {
                visibility: FeatureSwitches::load_string(SCALA_RULES_COPY).unwrap(),
                limited_actions: FeatureSwitches::load_string(SCALA_POLICY_COPY).unwrap(),
            })),
            copy,
        )
    }

    pub fn resolve(&self, viewer: &RequestViewer) -> ClientCapability {
        let default = ClientCapability::DEFAULT;
        if viewer.context().is_none() {
            return default;
        }
        let results = self.matched(viewer);
        let on =
            |key: &str, default_on: bool| results.get_bool_no_impression(key).unwrap_or(default_on);
        ClientCapability {
            verify_blur_support: verify_blur_support(|support| {
                on(
                    age_verification_key(support),
                    default.verify_blur_support == Some(support),
                )
            }),
            switches_on: ClientSwitch::VARIANTS
                .iter()
                .copied()
                .filter(|&switch| on(key(switch), default.has(switch)))
                .collect(),
        }
    }

    pub fn soft_nsfw_new_user_drop(&self, viewer: &RequestViewer) -> bool {
        viewer.id().is_some()
            && self
                .matched(viewer)
                .get_bool(SOFT_NSFW_NEW_USER_DROP)
                .unwrap_or(false)
    }

    pub fn experiments(&self, wired: ExperimentSet, viewer: &RequestViewer) -> RequestExperiments {
        if viewer.id().is_none() || wired.is_empty() {
            return RequestExperiments::default();
        }
        let results = self.matched(viewer);
        RequestExperiments::resolved(
            wired
                .iter()
                .filter(|experiment| {
                    experiment.is_on(results.get_bool_no_impression(experiment.key()))
                })
                .collect(),
        )
    }

    pub fn log_exposures(&self, facts: &RequestFacts) {
        let exposed = facts.experiments().exposed();
        if facts.viewer_id().is_none() || exposed.is_empty() {
            return;
        }
        let results = self.matched(facts.viewer());
        for experiment in exposed.iter() {
            results.get_bool(experiment.key());
        }
    }

    pub fn limited_actions_policies(
        &self,
        viewer: &RequestViewer,
        reasons: impl IntoIterator<Item = LimitedEngagementReason>,
        metrics: &Metrics,
    ) -> LimitedActionsPolicies {
        LimitedActionsPolicies::resolve(
            &self.feature_switches.load().limited_actions,
            &self.copy,
            viewer,
            reasons,
            metrics,
        )
    }

    fn matched(&self, viewer: &RequestViewer) -> FeatureSwitchResults {
        self.feature_switches
            .load()
            .visibility
            .match_recipient(&ViewerRecipient::new(viewer))
    }
}

pub(super) struct ViewerRecipient<'a> {
    user_id: Option<u64>,
    country: Option<&'a str>,
    language: Option<&'a str>,
    client_version: Option<&'a str>,
    client_app_id: Option<i64>,
    flavors: ClientFlavors,
    reason: Option<&'static str>,
}

impl<'a> ViewerRecipient<'a> {
    pub(super) fn new(viewer: &'a RequestViewer) -> Self {
        let mut recipient = Self {
            user_id: viewer.id(),
            country: viewer.country_code().filter(|country| !country.is_empty()),
            language: None,
            client_version: None,
            client_app_id: None,
            flavors: ClientFlavors::default(),
            reason: None,
        };
        if let Some(context) = viewer.context() {
            let user_agent = UserAgent::parse(&context.user_agent);
            recipient.client_version = user_agent.as_ref().and_then(UserAgent::version);
            recipient.flavors = flavors(context.client_application_id, user_agent.as_ref());
            recipient.client_app_id =
                (context.client_application_id != 0).then_some(context.client_application_id);
        }
        recipient
    }

    pub(super) fn with_language(self, language: Option<&'a str>) -> Self {
        Self {
            language: language.filter(|language| !language.is_empty()),
            ..self
        }
    }

    pub(super) fn set_reason(&mut self, reason: LimitedEngagementReason) {
        self.reason = Some(reason.limited_actions_string());
    }
}

impl Recipient for ViewerRecipient<'_> {
    fn user_id(&self) -> Option<u64> {
        self.user_id
    }

    fn country_code(&self) -> Option<&str> {
        self.country
    }

    fn language_code(&self) -> Option<&str> {
        self.language
    }

    fn client_version(&self) -> Option<&str> {
        self.client_version
    }

    fn client_app_id(&self) -> Option<i64> {
        self.client_app_id
    }

    fn android_flavor(&self) -> Option<&str> {
        self.flavors.android
    }

    fn ios_flavor(&self) -> Option<&str> {
        self.flavors.ios
    }

    fn mac_flavor(&self) -> Option<&str> {
        self.flavors.mac
    }

    fn rweb_flavor(&self) -> Option<&str> {
        self.flavors.rweb
    }

    fn get_custom_string(&self, field: &str) -> Option<&str> {
        self.reason.filter(|_| field == REASON_FIELD)
    }
}

fn flavors(app_id: i64, user_agent: Option<&UserAgent<'_>>) -> ClientFlavors {
    let all = |on: bool| on.then_some("all");
    let client_name = user_agent.map(|user_agent| user_agent.client_name);
    ClientFlavors {
        ios: all(user_agent.and_then(|user_agent| user_agent.os_brand) == Some("iOS")),
        android: all(matches!(
            client_name,
            Some("TwitterAndroid" | "XChatAndroid")
        )),
        mac: all(client_name == Some("Twitter-Mac")),
        rweb: flavors_for_app_id(app_id).rweb,
    }
}

struct UserAgent<'a> {
    client_name: &'a str,
    version: &'a str,
    os_brand: Option<&'a str>,
}

impl<'a> UserAgent<'a> {
    fn parse(user_agent: &'a str) -> Option<Self> {
        let user_agent = user_agent
            .strip_prefix('"')
            .and_then(|quoted| quoted.strip_suffix('"'))
            .unwrap_or(user_agent);
        Self::parse_standard(user_agent).or_else(|| Self::parse_legacy(user_agent))
    }

    fn parse_standard(user_agent: &'a str) -> Option<Self> {
        let name_end = user_agent
            .find(['/', '(', ')', ':'])
            .unwrap_or(user_agent.len());
        let (name, tail) = user_agent.split_at(name_end);
        let (client_name, rest) = (tail.starts_with('/').then_some(name_end).into_iter())
            .chain(name.rmatch_indices(' ').map(|(at, _)| at))
            .find_map(|at| {
                let (client_name, rest) = user_agent.split_at(at);
                let rest = rest.strip_prefix(['/', ' '])?;
                (!client_name.is_empty()
                    && rest.starts_with(|c: char| !matches!(c, '(' | ')' | ' ')))
                .then_some((client_name, rest))
            })?;
        let (version, rest) = rest.split_at(rest.find([' ', '(', ')']).unwrap_or(rest.len()));
        let model = |rest: &'a str| {
            let (model, os) = rest.strip_prefix(' ')?.split_once('/')?;
            (!model.is_empty() && os.starts_with(|c: char| !c.is_whitespace())).then_some(model)
        };
        let os_brand = rest
            .strip_prefix(" (")
            .and_then(|code| code.split_once(')'))
            .filter(|(code, _)| !code.contains('('))
            .and_then(|(_, rest)| model(rest))
            .or_else(|| model(rest));
        Some(Self {
            client_name,
            version,
            os_brand,
        })
    }

    fn parse_legacy(user_agent: &'a str) -> Option<Self> {
        let (head, version) = user_agent.rsplit_once(' ')?;
        let client_name = head.strip_suffix(':')?;
        (!client_name.is_empty()
            && !client_name.contains(['/', '(', ')'])
            && !version.is_empty()
            && !version.contains(['(', ')']))
        .then_some(Self {
            client_name,
            version,
            os_brand: None,
        })
    }

    fn version(&self) -> Option<&'a str> {
        if self.client_name == "Mozilla" {
            return None;
        }
        let mut rest = self.version;
        version_number(&mut rest)?;
        let minor = dotted_version_number(&mut rest);
        let patch = minor && dotted_version_number(&mut rest);
        let version = self.version.get(..self.version.len() - rest.len())?;
        let _ios_build = patch && dotted_version_number(&mut rest);
        let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        rest = after_run(Some(rest.trim_start_matches('-')), is_word).unwrap_or(rest);
        rest = after_run(rest.strip_prefix(['-', '.']), is_word).unwrap_or(rest);
        rest = after_run(rest.strip_prefix('.'), |c: char| c.is_ascii_digit()).unwrap_or(rest);
        rest.is_empty().then_some(version)
    }
}

fn version_number(rest: &mut &str) -> Option<()> {
    let len = rest.bytes().take(9).take_while(u8::is_ascii_digit).count();
    *rest = rest.get(len..).filter(|_| len > 0)?;
    Some(())
}

fn dotted_version_number(rest: &mut &str) -> bool {
    let Some(mut tail) = rest.strip_prefix('.') else {
        return false;
    };
    let read = version_number(&mut tail).is_some();
    if read {
        *rest = tail;
    }
    read
}

fn after_run(text: Option<&str>, class: impl Fn(char) -> bool) -> Option<&str> {
    let text = text?;
    let tail = text.trim_start_matches(class);
    (tail.len() < text.len()).then_some(tail)
}

fn verify_blur_support(on: impl Fn(VerifyBlurSupport) -> bool) -> Option<VerifyBlurSupport> {
    VerifyBlurSupport::VARIANTS
        .iter()
        .copied()
        .find(|&support| on(support))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rules::fixtures::CLIENT_CLASSES;
    use xai_feature_switches::FeatureSwitches;
    use xai_twittercontext_proto::TwitterContextViewer;

    const SOFT_NSFW_DDG_FILE: (&str, &str) = (
        "soft_nsfw.yml",
        "soft_nsfw:\n  parameters:\n    soft_nsfw_tweet_label_new_user_rule_enabled: {type: boolean, default: false}\n  rules:\n  - query: \"[soft_nsfw_ddg_1 bucket_membership treatment]\"\n    values: {soft_nsfw_tweet_label_new_user_rule_enabled: true}\n",
    );

    #[test]
    fn a_viewer_in_the_ddg_treatment_gets_the_switch_and_one_impression() {
        let impressor = Arc::new(xai_feature_switches::SpyingBucketImpressor::new());
        let switches = ClientSwitches::in_ddg_arm(
            "soft_nsfw_ddg_1",
            "treatment",
            &[SOFT_NSFW_DDG_FILE],
            |store| crate::params::Experiments {
                store: Some(store),
                impressor: Arc::clone(&impressor) as _,
            },
        );

        assert!(switches
            .soft_nsfw_new_user_drop(&RequestViewer::new(Some(TEST_USER_ID), Some("us".into()))));
        assert_eq!(impressor.impression_count("soft_nsfw_ddg_1"), 1);
        assert!(!switches.soft_nsfw_new_user_drop(&RequestViewer::new(None, Some("us".into()))));
        assert_eq!(impressor.impression_count("soft_nsfw_ddg_1"), 1);
    }

    #[test]
    fn outside_prod_a_viewer_in_the_ddg_treatment_still_gets_the_switch() {
        let switches = ClientSwitches::in_ddg_arm(
            "soft_nsfw_ddg_1",
            "treatment",
            &[SOFT_NSFW_DDG_FILE],
            |store| crate::params::Experiments::without_impressions(Some(store)),
        );

        assert!(switches
            .soft_nsfw_new_user_drop(&RequestViewer::new(Some(TEST_USER_ID), Some("us".into()))));
    }

    #[test]
    fn a_viewer_gets_its_arms_without_an_impression_and_one_per_exposed_experiment() {
        use crate::rules::Experiment::{TestHoldback, TestLaunch};
        let impressor = Arc::new(xai_feature_switches::SpyingBucketImpressor::new());
        let switches = ClientSwitches::in_ddg_arm(
            "test_ddg_1",
            "treatment",
            &[(
                "rust_vf.yml",
                "rust_vf:\n  parameters:\n    rust_vf_test_launch_rule_enabled: {type: boolean, default: false}\n    rust_vf_test_holdback_rule_enabled: {type: boolean, default: true}\n  rules:\n  - query: \"[test_ddg_1 bucket_membership treatment]\"\n    values: {rust_vf_test_launch_rule_enabled: true, rust_vf_test_holdback_rule_enabled: false}\n",
            )],
            |store| crate::params::Experiments {
                store: Some(store),
                impressor: Arc::clone(&impressor) as _,
            },
        );
        let wired = ExperimentSet::NONE.with(TestLaunch).with(TestHoldback);

        let facts = RequestFacts::logged_in(TEST_USER_ID);
        let arms = switches.experiments(wired, facts.viewer());
        assert!(arms.applies(TestLaunch));
        assert!(!arms.applies(TestHoldback));
        assert_eq!(impressor.impression_count("test_ddg_1"), 0);
        arms.expose(TestHoldback);
        switches.log_exposures(&facts.with_experiments(arms));
        assert_eq!(impressor.impression_count("test_ddg_1"), 1);

        let logged_out = switches.experiments(wired, RequestFacts::default().viewer());
        assert!(!logged_out.applies(TestLaunch) && logged_out.applies(TestHoldback));
    }

    const TEST_USER_ID: u64 = 1_000_000_000_000_000_001;

    fn viewer(id: Option<u64>, context: TwitterContextViewer) -> RequestViewer {
        RequestViewer::with_context(id, Some(context))
    }

    fn context(app_id: i64, user_agent: &str) -> TwitterContextViewer {
        TwitterContextViewer {
            client_application_id: app_id,
            user_agent: user_agent.into(),
            ..TwitterContextViewer::default()
        }
    }

    #[test]
    fn client_version_is_major_minor_patch_of_an_app_user_agent() {
        let version = |user_agent: &str| {
            UserAgent::parse(user_agent)
                .and_then(|user_agent| user_agent.version())
                .map(|v| v.to_string())
        };
        let ios = |version: &str| format!("Twitter-iPhone/{version} iOS/17.0 (Apple;iPhone15,2)");
        assert_eq!(version(&ios("11.11.5")).as_deref(), Some("11.11.5"));
        assert_eq!(version(&ios("11.11.5.2")).as_deref(), Some("11.11.5"));
        assert_eq!(version(&ios("6.50-Enterprise")).as_deref(), Some("6.50"));
        assert_eq!(version(&ios("10.50-beta.671")).as_deref(), Some("10.50"));
        assert_eq!(version(&ios("11.11.0-rc-1-x")), None);
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
                switches.resolve(&viewer(Some(1), client)),
                class.capability,
                "{}",
                class.name
            );
        }

        let third_party = context(0, "ThirdPartyClient/2.0");
        assert_eq!(
            switches.resolve(&viewer(Some(TEST_USER_ID), third_party)),
            ClientCapability {
                verify_blur_support: Some(VerifyBlurSupport::Supported),
                ..ClientCapability::DEFAULT.with(ClientSwitch::ModernBlur)
            }
        );
        let class = |name: &str| {
            CLIENT_CLASSES
                .iter()
                .find(|class| class.name == name)
                .unwrap()
        };
        let app_id = |name: &str| class(name).app_id;
        let ios_outdated = class("ios_outdated");
        assert_eq!(
            switches.resolve(&viewer(Some(1), context(0, ios_outdated.user_agent))),
            ios_outdated.capability
        );
        let dalvik = "Dalvik/2.1.0 (Linux; U; Android 14; Pixel 7 Build/UQ1A.240205.004)";
        assert_eq!(
            switches.resolve(&viewer(Some(1), context(app_id("android_current"), dalvik))),
            class("third_party_client").capability
        );
        assert_eq!(
            switches.resolve(&viewer(Some(1), context(0, "/9.0.0 iOS/17.0"))),
            class("third_party_client").capability
        );
        for user_agent in [
            "TwitterAndroid: 9.82.0",
            "TwitterAndroid 9.82.0 (29820000-r-0)",
        ] {
            assert_eq!(
                switches.resolve(&viewer(
                    Some(1),
                    context(app_id("android_current"), user_agent)
                )),
                class("android_without_fosnr").capability,
                "{user_agent}"
            );
        }
        let ios = app_id("ios_current");
        for (version, stale_tweet_limits) in [("9.19.9", false), ("9.20.0", true)] {
            let client = context(ios, &format!("Twitter-iPhone/{version} iOS/17.0"));
            assert_eq!(
                switches
                    .resolve(&viewer(Some(1), client))
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
                    .resolve(&viewer(Some(1), context(app_id, user_agent)))
                    .has(ClientSwitch::CommunityViewerRemovedLimits),
                community_viewer_removed_limits,
                "{user_agent}"
            );
        }
        let web = &CLIENT_CLASSES[0];
        assert!(!switches
            .resolve(&viewer(None, context(web.app_id, web.user_agent)))
            .has(ClientSwitch::CommunityViewerRemovedLimits));
        assert_eq!(
            switches.resolve(&RequestViewer::new(Some(TEST_USER_ID), None)),
            ClientCapability::default()
        );
        let no_keys = ClientSwitches::new(
            Arc::new(ArcSwap::from_pointee(SwitchEngines {
                visibility: FeatureSwitches::null().unwrap(),
                limited_actions: FeatureSwitches::null().unwrap(),
            })),
            LimitedActionsCopy::from_json("[]"),
        );
        let web = &CLIENT_CLASSES[0];
        assert_eq!(
            no_keys.resolve(&viewer(None, context(web.app_id, web.user_agent))),
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
