use crate::models::{HydratedTweetCandidate, VfAction, ViewerFeatures};
use crate::params::NsfwGatingCountries;
use crate::rules::rule_spec::RuleSpec;
use crate::rules::{author_rules, tweet_rules};
use crate::rules::{RuleContext, Verdict};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafetyLevel {
    FilterAll,
    TimelineHome,
    TimelineHomeRecommendations,
}

impl SafetyLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            SafetyLevel::FilterAll => "filter_all",
            SafetyLevel::TimelineHome => "timeline_home",
            SafetyLevel::TimelineHomeRecommendations => "timeline_home_recommendations",
        }
    }
}

pub(super) struct Policy {
    rules: &'static [&'static [RuleSpec]],
    additional_rules: &'static [&'static [RuleSpec]],
}

impl Policy {
    pub(super) const fn new(rules: &'static [&'static [RuleSpec]]) -> Self {
        Self {
            rules,
            additional_rules: &[],
        }
    }

    fn rules(&self) -> impl Iterator<Item = &'static RuleSpec> + '_ {
        self.rules
            .iter()
            .chain(self.additional_rules)
            .copied()
            .flatten()
    }

    pub(super) fn evaluate(&self, context: &RuleContext<'_>) -> Verdict {
        let mut action = VfAction::Allow;
        let mut decided_by = None;

        for rule in self.rules() {
            match rule.evaluate(context) {
                VfAction::Drop(reason) => {
                    return Verdict {
                        action: VfAction::Drop(reason),
                        decided_by: Some(rule.name()),
                    };
                }
                VfAction::Interstitial(reason) if matches!(action, VfAction::Allow) => {
                    action = VfAction::Interstitial(reason);
                    decided_by = Some(rule.name());
                }
                VfAction::Allow | VfAction::Interstitial(_) => {}
            }
        }

        Verdict { action, decided_by }
    }

    fn len(&self) -> usize {
        self.rules().count()
    }

    #[cfg(test)]
    fn rule_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.rules().map(RuleSpec::name)
    }
}

static FILTER_ALL_POLICY: Policy = Policy::new(&[tweet_rules::FILTER_ALL]);

static TIMELINE_HOME_SHARED_RULES: [&[RuleSpec]; 9] = [
    author_rules::AUTHOR_STATE_DROPS,
    author_rules::SOCIALGRAPH_DROPS,
    tweet_rules::TWEET_LABEL_DROPS,
    tweet_rules::NULLCAST_DROP,
    tweet_rules::TES_HOME_DROPS,
    tweet_rules::SENSITIVE_VIEWER_DROPS,
    tweet_rules::EXCLUSIVE_TWEET_DROP,
    tweet_rules::NSFW_MEDIA_INTERSTITIALS,
    tweet_rules::NSFW_AUTHOR_INTERSTITIAL,
];

static TIMELINE_HOME_RECOMMENDATION_ONLY_RULES: [&[RuleSpec]; 5] = [
    tweet_rules::RECS_MEDIA_DROPS,
    author_rules::OON_NSFW_AUTHOR_DROPS,
    tweet_rules::OON_TWEET_FLAG_DROPS,
    tweet_rules::OON_TWEET_LABEL_DROPS,
    author_rules::OON_USER_LABEL_DROPS,
];

static TIMELINE_HOME_POLICY: Policy = Policy::new(&TIMELINE_HOME_SHARED_RULES);
static TIMELINE_HOME_RECOMMENDATIONS_POLICY: Policy = Policy {
    rules: &TIMELINE_HOME_SHARED_RULES,
    additional_rules: &TIMELINE_HOME_RECOMMENDATION_ONLY_RULES,
};

pub struct RuleEngine {
    nsfw_gating_countries: Arc<NsfwGatingCountries>,
}

impl RuleEngine {
    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Self::with_nsfw_gating_countries(Arc::new(NsfwGatingCountries::starting_at_default()))
    }

    pub fn with_nsfw_gating_countries(gating_countries: Arc<NsfwGatingCountries>) -> Self {
        Self {
            nsfw_gating_countries: gating_countries,
        }
    }

    fn select(level: SafetyLevel) -> &'static Policy {
        match level {
            SafetyLevel::FilterAll => &FILTER_ALL_POLICY,
            SafetyLevel::TimelineHome => &TIMELINE_HOME_POLICY,
            SafetyLevel::TimelineHomeRecommendations => &TIMELINE_HOME_RECOMMENDATIONS_POLICY,
        }
    }

    pub fn evaluate(
        &self,
        level: SafetyLevel,
        viewer: &ViewerFeatures,
        candidate: &HydratedTweetCandidate,
    ) -> Verdict {
        let context = RuleContext::new(viewer, candidate, &self.nsfw_gating_countries);
        Self::select(level).evaluate(&context)
    }

    #[cfg(test)]
    pub(crate) fn wired_rule_names(&self, level: SafetyLevel) -> Vec<&'static str> {
        Self::select(level).rule_names().collect()
    }

    pub fn rule_counts(&self) -> (usize, usize) {
        (
            TIMELINE_HOME_POLICY.len(),
            TIMELINE_HOME_RECOMMENDATIONS_POLICY.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{VfAction, ViewerFeatures};
    use crate::rules::fixtures::{candidate, viewer, VIEWER_ID};

    #[test]
    fn refreshed_config_country_reaches_the_wired_rule() {
        let gating_countries = Arc::new(NsfwGatingCountries::starting_at_default());
        let rule_engine = RuleEngine::with_nsfw_gating_countries(Arc::clone(&gating_countries));
        let candidate = candidate()
            .with_label(crate::models::SafetyLabelType::NSFW_HIGH_PRECISION)
            .with_media()
            .build();
        let viewer = ViewerFeatures {
            viewer_age: crate::models::ViewerAge::NotStated,
            country_code: Some("us".into()),
            ..viewer(VIEWER_ID)
        };

        let verdict = rule_engine.evaluate(SafetyLevel::TimelineHome, &viewer, &candidate);
        assert!(!matches!(verdict.action, VfAction::Drop(_)));

        gating_countries.refresh_from(
            &xai_feature_switches::FeatureSwitches::load_string(
                r#"
rust_vf:
  parameters:
    rust_vf_nsfw_gating_countries:
      type: array
      default:
      - "us"
"#,
            )
            .unwrap(),
        );
        let verdict = rule_engine.evaluate(SafetyLevel::TimelineHome, &viewer, &candidate);
        assert!(matches!(verdict.action, VfAction::Drop(_)));
        assert_eq!(
            verdict.decided_by,
            Some("SensitiveViewerNoStatedAgeDropRule")
        );
    }

    #[test]
    fn wired_rule_order_matches_pre_migration_sequence() {
        let rule_engine = RuleEngine::for_tests();
        assert_eq!(
            rule_engine.wired_rule_names(SafetyLevel::FilterAll),
            vec!["FilterAllRule"]
        );
        let home = rule_engine.wired_rule_names(SafetyLevel::TimelineHome);
        assert_eq!(
            home,
            vec![
                "SuspendedAuthorRule",
                "DeactivatedAuthorRule",
                "ErasedAuthorRule",
                "OffboardedAuthorRule",
                "ProtectedAuthorDropRule",
                "ViewerBlocksAuthorRule",
                "ViewerMutesAuthorRule",
                "MutedRetweetsRule",
                "PdnaTweetLabelRule",
                "BounceTweetLabelRule",
                "SpamTweetLabelRule",
                "ForEmergencyUseOnlyDropRule",
                "FosnrHatefulConductDropRule",
                "FosnrViolentSpeechDropRule",
                "FosnrAbuseDropRule",
                "NullcastedTweetDropRule",
                "DropStaleTweetsRule",
                "DropLegalTakendownPostRule",
                "DropLocalLawsTakendownPostRule",
                "SensitiveViewerLoggedOutDropRule",
                "SensitiveViewerUnderageDropRule",
                "SensitiveViewerNoStatedAgeDropRule",
                "DropExclusiveTweetContentRule",
                "NsfwHighPrecisionInterstitialRule",
                "GoreAndViolenceInterstitialRule",
                "NsfwCardImageInterstitialRule",
                "NsfwAuthorInterstitialRule",
            ]
        );
        let mut recs = home.clone();
        recs.extend([
            "DropTweetsWithDmcaMediaRule",
            "DropTweetsWithGeoRestrictedMediaRule",
            "DropNsfwUserAuthorRule",
            "DropNsfwAdminAuthorRule",
            "TweetNsfwUserDropRule",
            "TweetNsfwAdminDropRule",
            "NsfwHighRecallDropRule",
            "NsfwHighPrecisionOonDropRule",
            "GoreAndViolenceOonDropRule",
            "NsfwCardImageOonDropRule",
            "DoNotAmplifyOonDropRule",
            "FosnrCivicIntegrityDropRule",
            "MaliciousUrlOonDropRule",
            "SpamHighRecallDropRule",
            "NsfwTextTweetLabelDropRule",
            "FosnrAbuseInsultsOonDropRule",
            "NsfwHighRecallUserLabelRule",
            "NsfwHighPrecisionUserLabelRule",
            "SpamHighRecallUserLabelRule",
            "CompromisedUserLabelRule",
            "ReadOnlyUserLabelRule",
            "ImpersonationHighPrecisionUserLabelRule",
            "NsfwAvatarImageRule",
            "NsfwBannerImageRule",
            "AbusiveHighRecallRule",
            "NsfwNearPerfectAuthorRule",
            "DoNotAmplifyNonFollowerRule",
        ]);
        assert_eq!(
            rule_engine.wired_rule_names(SafetyLevel::TimelineHomeRecommendations),
            recs
        );
    }

    #[derive(Debug, PartialEq, Eq)]
    enum RowClass {
        Drop,
        Interstitial,
    }

    fn row_class(spec: &RuleSpec) -> RowClass {
        use crate::rules::rule_spec::RuleAction;
        match spec {
            RuleSpec::Tweet {
                action: RuleAction::Drop(_),
                ..
            }
            | RuleSpec::Author { .. }
            | RuleSpec::Custom { .. } => RowClass::Drop,
            RuleSpec::Tweet {
                action: RuleAction::SensitiveMediaInterstitial(_),
                ..
            } => RowClass::Interstitial,
        }
    }

    fn assert_drops_precede_interstitials(policy: &Policy, name: &str) {
        let mut seen_interstitial = false;
        for spec in policy.rules() {
            match row_class(spec) {
                RowClass::Interstitial => seen_interstitial = true,
                RowClass::Drop if seen_interstitial => {
                    panic!("{name}: drop {} follows an interstitial row", spec.name());
                }
                RowClass::Drop => {}
            }
        }
    }

    #[test]
    fn drops_precede_interstitials_and_recommendation_only_rules_are_drops() {
        assert_drops_precede_interstitials(&FILTER_ALL_POLICY, "FilterAll");
        assert_drops_precede_interstitials(&TIMELINE_HOME_POLICY, "TimelineHome");
        for spec in TIMELINE_HOME_RECOMMENDATION_ONLY_RULES
            .iter()
            .copied()
            .flatten()
        {
            assert_eq!(
                row_class(spec),
                RowClass::Drop,
                "recommendation-only row {} must be a drop",
                spec.name()
            );
        }
    }

    mod engine {
        use super::super::*;
        use crate::models::{HydratedTweetCandidate, VfAction, ViewerFeatures};
        use crate::rules::test_context;
        use xai_visibility_filtering::models::FilteredReason;

        const fn custom(
            name: &'static str,
            evaluate: fn(&RuleContext<'_>) -> VfAction,
        ) -> RuleSpec {
            RuleSpec::Custom { name, evaluate }
        }

        fn allow(_: &RuleContext<'_>) -> VfAction {
            VfAction::Allow
        }

        fn drop_suspended(_: &RuleContext<'_>) -> VfAction {
            VfAction::Drop(FilteredReason::AuthorIsSuspended)
        }

        fn interstitial_nsfw(_: &RuleContext<'_>) -> VfAction {
            VfAction::Interstitial(FilteredReason::ContainNsfwMedia)
        }

        fn interstitial_unspecified(_: &RuleContext<'_>) -> VfAction {
            VfAction::Interstitial(FilteredReason::UnspecifiedReason)
        }

        fn unreachable_after_drop(_: &RuleContext<'_>) -> VfAction {
            panic!("a rule after a Drop must never be evaluated");
        }

        fn context_inputs() -> (ViewerFeatures, HydratedTweetCandidate) {
            (ViewerFeatures::default(), HydratedTweetCandidate::default())
        }

        static SHORT_CIRCUIT_ROWS: [RuleSpec; 3] = [
            custom("allow", allow),
            custom("drop", drop_suspended),
            custom("after_drop", unreachable_after_drop),
        ];
        static SHORT_CIRCUIT: Policy = Policy::new(&[&SHORT_CIRCUIT_ROWS]);

        static INTERSTITIAL_ROWS: [RuleSpec; 2] = [
            custom("first_interstitial", interstitial_nsfw),
            custom("second_interstitial", interstitial_unspecified),
        ];
        static INTERSTITIALS: Policy = Policy::new(&[&INTERSTITIAL_ROWS]);

        static DROP_AFTER_INTERSTITIAL_ROWS: [RuleSpec; 2] = [
            custom("interstitial", interstitial_nsfw),
            custom("drop", drop_suspended),
        ];
        static DROP_AFTER_INTERSTITIAL: Policy = Policy::new(&[&DROP_AFTER_INTERSTITIAL_ROWS]);

        static ALL_ALLOWS_ROWS: [RuleSpec; 2] = [custom("a", allow), custom("b", allow)];
        static ALL_ALLOWS: Policy = Policy::new(&[&ALL_ALLOWS_ROWS]);

        #[test]
        fn drop_short_circuits_later_rules() {
            let (viewer, candidate) = context_inputs();

            let verdict = SHORT_CIRCUIT.evaluate(&test_context(&viewer, &candidate));

            assert!(matches!(
                verdict.action,
                VfAction::Drop(FilteredReason::AuthorIsSuspended)
            ));
            assert_eq!(verdict.decided_by, Some("drop"));
        }

        #[test]
        fn first_interstitial_sticks_without_short_circuit() {
            let (viewer, candidate) = context_inputs();

            let verdict = INTERSTITIALS.evaluate(&test_context(&viewer, &candidate));

            assert!(matches!(
                verdict.action,
                VfAction::Interstitial(FilteredReason::ContainNsfwMedia)
            ));
            assert_eq!(verdict.decided_by, Some("first_interstitial"));
        }

        #[test]
        fn drop_after_interstitial_wins() {
            let (viewer, candidate) = context_inputs();

            let verdict = DROP_AFTER_INTERSTITIAL.evaluate(&test_context(&viewer, &candidate));

            assert!(matches!(
                verdict.action,
                VfAction::Drop(FilteredReason::AuthorIsSuspended)
            ));
            assert_eq!(verdict.decided_by, Some("drop"));
        }

        #[test]
        fn all_allows_is_allow_with_no_decider() {
            let (viewer, candidate) = context_inputs();

            let verdict = ALL_ALLOWS.evaluate(&test_context(&viewer, &candidate));

            assert!(matches!(verdict.action, VfAction::Allow));
            assert_eq!(verdict.decided_by, None);
        }
    }
}
