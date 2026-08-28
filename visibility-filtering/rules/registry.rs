use crate::models::{HydratedTweetCandidate, VfAction, ViewerFeatures};
use crate::rules::nsfw_age_gating::{
    SensitiveViewerLoggedOutDropRule, SensitiveViewerNoStatedAgeDropRule,
    SensitiveViewerUnderageDropRule,
};
use crate::rules::nsfw_interstitial::{
    NsfwAuthorInterstitialRule, GORE_AND_VIOLENCE_INTERSTITIAL, NSFW_CARD_IMAGE_INTERSTITIAL,
    NSFW_HIGH_PRECISION_INTERSTITIAL,
};
use crate::rules::nullcast_rule::NullcastedTweetDropRule;
use crate::rules::socialgraph_rules::{
    DropExclusiveTweetContentRule, MutedRetweetsRule, ViewerBlocksAuthorRule, ViewerMutesAuthorRule,
};
use crate::rules::tes_rules::{
    DropLegalTakendownPostRule, DropLocalLawsTakendownPostRule, DropStaleTweetsRule,
    DropTweetsWithDmcaMediaRule, DropTweetsWithGeoRestrictedMediaRule,
};
use crate::rules::tweet_flag_rules as tweet_flag;
use crate::rules::tweet_label_drops as tweet_label;
use crate::rules::user_label_drops as user_label;
use crate::rules::user_rules::{self as author, ProtectedAuthorDropRule};
use crate::rules::{evaluate_rules, Rule, RuleContext, Verdict};
use crate::rules::high_risk_tag_drops;
use xai_visibility_filtering::models::FilteredReason;



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

pub struct Policies {
    filter_all: Vec<Box<dyn Rule>>,
    timeline_home: Vec<Box<dyn Rule>>,
    timeline_home_recommendations: Vec<Box<dyn Rule>>,
}

impl Policies {
    pub fn new() -> Self {
        Self {
            filter_all: vec![Box::new(FilterAllRule)],
            timeline_home: timeline_home_policy(),
            timeline_home_recommendations: timeline_home_recommendations_policy(),
        }
    }

    fn select(&self, level: SafetyLevel) -> &[Box<dyn Rule>] {
        match level {
            SafetyLevel::FilterAll => &self.filter_all,
            SafetyLevel::TimelineHome => &self.timeline_home,
            SafetyLevel::TimelineHomeRecommendations => &self.timeline_home_recommendations,
        }
    }

    pub fn evaluate(
        &self,
        level: SafetyLevel,
        viewer: &ViewerFeatures,
        candidate: &HydratedTweetCandidate,
    ) -> Verdict {
        let context = RuleContext::new(level, viewer, candidate);
        evaluate_rules(self.select(level), &context)
    }

    pub fn rule_counts(&self) -> (usize, usize) {
        (
            self.timeline_home.len(),
            self.timeline_home_recommendations.len(),
        )
    }
}

impl Default for Policies {
    fn default() -> Self {
        Self::new()
    }
}

struct FilterAllRule;

impl Rule for FilterAllRule {
    fn name(&self) -> &'static str {
        "FilterAllRule"
    }

    fn evaluate(&self, _context: &RuleContext<'_>) -> VfAction {
        VfAction::Drop(FilteredReason::UnspecifiedReason)
    }
}

fn base_home_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(author::SUSPENDED_AUTHOR_DROP),
        Box::new(author::DEACTIVATED_AUTHOR_DROP),
        Box::new(author::ERASED_AUTHOR_DROP),
        Box::new(author::OFFBOARDED_AUTHOR_DROP),
        Box::new(ProtectedAuthorDropRule),
        Box::new(ViewerBlocksAuthorRule),
        Box::new(ViewerMutesAuthorRule),
        Box::new(MutedRetweetsRule),
        Box::new(tweet_label::PDNA_DROP),
        Box::new(tweet_label::BOUNCE_DROP),
        // ← ADD THIS LINE
        Box::new(high_risk_tag_drops::HIGH_RISK_TAG_DROP),   // or whatever you named the const
        
        Box::new(tweet_label::SPAM_DROP),
        Box::new(tweet_label::FOR_EMERGENCY_USE_ONLY_DROP),
        Box::new(tweet_label::FOSNR_HATEFUL_CONDUCT_DROP),
        Box::new(tweet_label::FOSNR_VIOLENT_SPEECH_DROP),
        Box::new(tweet_label::FOSNR_ABUSE_DROP),
        Box::new(tweet_label::FOSNR_CIVIC_INTEGRITY_DROP),
        Box::new(NullcastedTweetDropRule),
        Box::new(DropStaleTweetsRule),
        Box::new(DropLegalTakendownPostRule),
        Box::new(DropLocalLawsTakendownPostRule),
        Box::new(SensitiveViewerLoggedOutDropRule),
        Box::new(SensitiveViewerUnderageDropRule),
        Box::new(SensitiveViewerNoStatedAgeDropRule),
        Box::new(DropExclusiveTweetContentRule),
        Box::new(NSFW_HIGH_PRECISION_INTERSTITIAL),
        Box::new(GORE_AND_VIOLENCE_INTERSTITIAL),
        Box::new(NSFW_CARD_IMAGE_INTERSTITIAL),
        Box::new(NsfwAuthorInterstitialRule),
    ]
}

fn timeline_home_policy() -> Vec<Box<dyn Rule>> {
    base_home_rules()
}

fn timeline_home_recommendations_policy() -> Vec<Box<dyn Rule>> {
    let mut rules = base_home_rules();
    let oon_drops: Vec<Box<dyn Rule>> = vec![
        Box::new(DropTweetsWithDmcaMediaRule),
        Box::new(DropTweetsWithGeoRestrictedMediaRule),
        Box::new(author::NSFW_USER_AUTHOR_DROP),
        Box::new(author::NSFW_ADMIN_AUTHOR_DROP),
        Box::new(tweet_flag::TWEET_NSFW_USER_DROP),
        Box::new(tweet_flag::TWEET_NSFW_ADMIN_DROP),
        Box::new(tweet_label::NSFW_HIGH_RECALL_DROP),
        Box::new(tweet_label::NSFW_HIGH_PRECISION_DROP),
        Box::new(tweet_label::GORE_AND_VIOLENCE_HIGH_PRECISION_DROP),
        Box::new(tweet_label::NSFW_CARD_IMAGE_DROP),
        Box::new(tweet_label::DO_NOT_AMPLIFY_DROP),
        Box::new(tweet_label::MALICIOUS_URL_DROP),
        Box::new(tweet_label::SPAM_HIGH_RECALL_DROP),
        Box::new(tweet_label::NSFW_TEXT_DROP),
        Box::new(tweet_label::FOSNR_ABUSE_INSULTS_OON_DROP),
        Box::new(user_label::NSFW_HIGH_RECALL_USER_DROP),
        Box::new(user_label::NSFW_HIGH_PRECISION_USER_DROP),
        Box::new(user_label::SPAM_HIGH_RECALL_USER_DROP),
        Box::new(user_label::COMPROMISED_USER_DROP),
        Box::new(user_label::READ_ONLY_USER_DROP),
        Box::new(user_label::IMPERSONATION_HIGH_PRECISION_USER_DROP),
        Box::new(user_label::NSFW_AVATAR_IMAGE_USER_DROP),
        Box::new(user_label::NSFW_BANNER_IMAGE_USER_DROP),
        Box::new(user_label::ABUSIVE_HIGH_RECALL_USER_DROP),
        Box::new(user_label::NSFW_NEAR_PERFECT_USER_DROP),
        Box::new(user_label::DO_NOT_AMPLIFY_NON_FOLLOWER_USER_DROP),
    ];
    rules.extend(oon_drops);
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        HydratedTweetCandidate, MediaFeature, TweetFeatures, Viewer, ViewerFeatures,
    };
    use std::collections::HashMap;

    struct RecommendationsOnlyRule;

    impl Rule for RecommendationsOnlyRule {
        fn name(&self) -> &'static str {
            "RecommendationsOnlyRule"
        }

        fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
            match context.safety_level() {
                SafetyLevel::TimelineHomeRecommendations => {
                    VfAction::Drop(FilteredReason::UnspecifiedReason)
                }
                SafetyLevel::FilterAll | SafetyLevel::TimelineHome => VfAction::Allow,
            }
        }
    }

    #[test]
    fn policies_evaluate_uses_selected_safety_level_in_context() {
        let policies = Policies {
            filter_all: vec![Box::new(RecommendationsOnlyRule)],
            timeline_home: vec![Box::new(RecommendationsOnlyRule)],
            timeline_home_recommendations: vec![Box::new(RecommendationsOnlyRule)],
        };
        let viewer = ViewerFeatures::default();
        let candidate = HydratedTweetCandidate::default();

        assert!(matches!(
            policies
                .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
                .action,
            VfAction::Allow
        ));
        assert!(matches!(
            policies
                .evaluate(
                    SafetyLevel::TimelineHomeRecommendations,
                    &viewer,
                    &candidate
                )
                .action,
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn filter_all_rule_drops_even_self_view() {
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(100),
            ..Default::default()
        };
        assert!(matches!(
            FilterAllRule.evaluate(&crate::rules::test_context(&viewer, &candidate)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn filter_all_policy_drops_pristine_candidate() {
        let policies = Policies::new();
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            ..Default::default()
        };
        let verdict = policies.evaluate(
            SafetyLevel::FilterAll,
            &ViewerFeatures::default(),
            &candidate,
        );
        assert!(matches!(verdict.action, VfAction::Drop(_)));

        let verdict = policies.evaluate(
            SafetyLevel::TimelineHome,
            &ViewerFeatures::default(),
            &candidate,
        );
        assert!(matches!(verdict.action, VfAction::Allow));
    }

    #[test]
    fn dmca_media_drops_recommendations_only() {
        let policies = Policies::new();
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            tweet_features: TweetFeatures {
                media: MediaFeature {
                    has_dmca_media: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };

        let timeline_home = policies.evaluate(
            SafetyLevel::TimelineHome,
            &ViewerFeatures::default(),
            &candidate,
        );
        assert!(matches!(timeline_home.action, VfAction::Allow));

        let recommendations = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &ViewerFeatures::default(),
            &candidate,
        );
        assert!(matches!(recommendations.action, VfAction::Drop(_)));
    }

    #[test]
    fn tweet_nsfw_flag_drops_recommendations_only() {
        use crate::models::NsfwFeature;
        let policies = Policies::new();
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            tweet_features: TweetFeatures {
                nsfw: NsfwFeature {
                    user: true,
                    admin: false,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let timeline_home = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(timeline_home, VfAction::Allow),
            "in-network tweet nsfw_user flag should allow, got {timeline_home:?}"
        );

        let recommendations = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &candidate,
        );
        assert!(matches!(recommendations.action, VfAction::Drop(_)));
        assert_eq!(recommendations.decided_by, Some("TweetNsfwUserDropRule"));
    }

    #[test]
    fn nsfw_author_interstitials_in_network_but_drops_oon() {
        use crate::models::{AuthorFeatures, TweetFeatures};
        let policies = Policies::new();
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            author_features: AuthorFeatures {
                is_nsfw_user: true,
                ..Default::default()
            },
            tweet_features: TweetFeatures {
                media: MediaFeature {
                    has_media: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            allows_sensitive_media: false,
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Interstitial(_)),
            "in-network NSFW author should interstitial, got {in_network:?}"
        );

        let oon = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &viewer,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon, VfAction::Drop(_)),
            "OON NSFW author should drop, got {oon:?}"
        );
    }

    #[test]
    fn egregious_nsfw_does_not_drop() {
        use crate::models::{SafetyLabel, SafetyLabelMap, SafetyLabelType};
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();

        let mut labels = HashMap::new();
        labels.insert(SafetyLabelType::EGREGIOUS_NSFW, SafetyLabel::default());
        let tweet_candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            ..Default::default()
        };
        let user_candidate = candidate_with_author_user_label(LabelValue::EGREGIOUS_NSFW, false);
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        for candidate in [&tweet_candidate, &user_candidate] {
            let in_network = policies
                .evaluate(SafetyLevel::TimelineHome, &viewer, candidate)
                .action;
            assert!(
                matches!(in_network, VfAction::Allow),
                "in-network EgregiousNsfw should allow after rule removal, got {in_network:?}"
            );
            let oon = policies
                .evaluate(SafetyLevel::TimelineHomeRecommendations, &viewer, candidate)
                .action;
            assert!(
                matches!(oon, VfAction::Allow),
                "OON EgregiousNsfw should allow after rule removal, got {oon:?}"
            );
        }
    }

    #[test]
    fn spam_high_recall_drops_oon_but_allows_in_network() {
        use crate::models::{SafetyLabel, SafetyLabelMap, SafetyLabelType};
        let policies = Policies::new();
        let mut labels = HashMap::new();
        labels.insert(SafetyLabelType::SPAM_HIGH_RECALL, SafetyLabel::default());
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network SpamHighRecall should allow (Scala drops it OON only), got {in_network:?}"
        );

        let oon = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &viewer,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon, VfAction::Drop(_)),
            "OON SpamHighRecall should drop, got {oon:?}"
        );
    }

    #[test]
    fn malicious_url_drops_oon_but_allows_in_network() {
        use crate::models::{SafetyLabel, SafetyLabelMap, SafetyLabelType};
        let policies = Policies::new();
        let mut labels = HashMap::new();
        labels.insert(SafetyLabelType::MALICIOUS_URL, SafetyLabel::default());
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network MaliciousUrl should allow (Scala drops it OON only), got {in_network:?}"
        );

        let oon = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &candidate,
        );
        assert!(
            matches!(oon.action, VfAction::Drop(_)),
            "OON MaliciousUrl should drop, got {:?}",
            oon.action
        );
        assert_eq!(oon.decided_by, Some("MaliciousUrlOonDropRule"));

        let author = ViewerFeatures {
            viewer: Viewer::LoggedIn(100),
            ..Default::default()
        };
        let oon_author = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &author,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon_author, VfAction::Allow),
            "OON MaliciousUrl should allow author, got {oon_author:?}"
        );
    }

    fn fosnr_candidate(
        label: crate::models::SafetyLabelType,
        follows: bool,
    ) -> HydratedTweetCandidate {
        use crate::models::{SafetyLabel, SafetyLabelMap};
        let mut labels = HashMap::new();
        labels.insert(label, SafetyLabel::default());
        let mut c = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            ..Default::default()
        };
        c.relationship.viewer_follows_author = follows;
        c
    }

    #[test]
    fn fosnr_labels_drop_non_author_non_follower_on_both_surfaces() {
        use crate::models::SafetyLabelType;
        let policies = Policies::new();
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };
        for label in [
            SafetyLabelType::FOSNR_HATEFUL_CONDUCT,
            SafetyLabelType::FOSNR_VIOLENT_SPEECH,
            SafetyLabelType::FOSNR_ABUSE,
            SafetyLabelType::FOSNR_CIVIC_INTEGRITY,
        ] {
            let candidate = fosnr_candidate(label, false);
            for level in [
                SafetyLevel::TimelineHome,
                SafetyLevel::TimelineHomeRecommendations,
            ] {
                let action = policies.evaluate(level, &viewer, &candidate).action;
                assert!(
                    matches!(action, VfAction::Drop(_)),
                    "{label:?} on {level:?} should drop non-follower, got {action:?}"
                );
            }
        }
    }

    #[test]
    fn fosnr_never_drops_author() {
        use crate::models::SafetyLabelType;
        let policies = Policies::new();
        let author = ViewerFeatures {
            viewer: Viewer::LoggedIn(100),
            ..Default::default()
        };
        for label in [
            SafetyLabelType::FOSNR_HATEFUL_CONDUCT,
            SafetyLabelType::FOSNR_VIOLENT_SPEECH,
            SafetyLabelType::FOSNR_ABUSE,
            SafetyLabelType::FOSNR_CIVIC_INTEGRITY,
            SafetyLabelType::FOSNR_ABUSE_INSULTS,
        ] {
            let candidate = fosnr_candidate(label, false);
            for level in [
                SafetyLevel::TimelineHome,
                SafetyLevel::TimelineHomeRecommendations,
            ] {
                let action = policies.evaluate(level, &author, &candidate).action;
                assert!(
                    matches!(action, VfAction::Allow),
                    "{label:?} on {level:?} should allow author, got {action:?}"
                );
            }
        }
    }

    #[test]
    fn fosnr_abuse_insults_drops_oon_but_allows_in_network() {
        use crate::models::SafetyLabelType;
        let policies = Policies::new();
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };
        let author = ViewerFeatures {
            viewer: Viewer::LoggedIn(100),
            ..Default::default()
        };

        for follows in [true, false] {
            let candidate = fosnr_candidate(SafetyLabelType::FOSNR_ABUSE_INSULTS, follows);
            let in_network = policies
                .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
                .action;
            assert!(
                matches!(in_network, VfAction::Allow),
                "in-network FosnrAbuseInsults should allow (follows={follows}), got {in_network:?}"
            );
        }

        let candidate = fosnr_candidate(SafetyLabelType::FOSNR_ABUSE_INSULTS, false);
        let oon = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &viewer,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon, VfAction::Drop(_)),
            "OON FosnrAbuseInsults should drop non-author, got {oon:?}"
        );

        let oon_author = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &author,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon_author, VfAction::Allow),
            "OON FosnrAbuseInsults should allow author, got {oon_author:?}"
        );
    }

    #[test]
    fn geo_restricted_media_drops_oon_but_allows_in_network() {
        use crate::models::TweetFeatures;
        let policies = Policies::new();
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            tweet_features: TweetFeatures {
                media: MediaFeature {
                    geo_deny_list: vec!["de".to_string()],
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            country_code: Some("de".to_string()),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network geo-restricted media should allow (Scala wires the rule in THR only), got {in_network:?}"
        );

        let oon = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &candidate,
        );
        assert!(
            matches!(oon.action, VfAction::Drop(_)),
            "OON geo-restricted media should drop, got {:?}",
            oon.action
        );
        assert_eq!(oon.decided_by, Some("DropTweetsWithGeoRestrictedMediaRule"));
    }

    #[test]
    fn nsfw_text_drops_oon_but_allows_in_network() {
        use crate::models::{SafetyLabel, SafetyLabelMap, SafetyLabelType};
        let policies = Policies::new();
        let mut labels = HashMap::new();
        labels.insert(SafetyLabelType::NSFW_TEXT, SafetyLabel::default());
        let candidate = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            ..Default::default()
        };
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network NsfwText should allow (Scala drops it OON only), got {in_network:?}"
        );

        let oon = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &viewer,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon, VfAction::Drop(_)),
            "OON NsfwText should drop, got {oon:?}"
        );
    }

    fn candidate_with_author_user_label(
        label: xai_x_thrift::user_labels::LabelValue,
        follows: bool,
    ) -> HydratedTweetCandidate {
        use crate::models::{AuthorFeatures, UserLabelSet};
        use std::collections::HashSet;
        let mut c = HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            author_features: AuthorFeatures {
                user_labels: UserLabelSet::new(HashSet::from([label])),
                ..Default::default()
            },
            ..Default::default()
        };
        c.relationship.viewer_follows_author = follows;
        c
    }

    #[test]
    fn nsfw_avatar_user_label_drops_oon_but_allows_in_network() {
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();
        let candidate = candidate_with_author_user_label(LabelValue::NSFW_AVATAR_IMAGE, false);
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network NsfwAvatarImage should allow, got {in_network:?}"
        );

        let oon = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &candidate,
        );
        assert!(
            matches!(oon.action, VfAction::Drop(_)),
            "OON NsfwAvatarImage should drop, got {:?}",
            oon.action
        );
        assert_eq!(oon.decided_by, Some("NsfwAvatarImageRule"));
    }

    #[test]
    fn recommendations_blacklist_does_not_drop() {
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();
        let candidate =
            candidate_with_author_user_label(LabelValue::RECOMMENDATIONS_BLACKLIST, false);
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network RecommendationsBlacklist should allow, got {in_network:?}"
        );

        let oon = policies
            .evaluate(
                SafetyLevel::TimelineHomeRecommendations,
                &viewer,
                &candidate,
            )
            .action;
        assert!(
            matches!(oon, VfAction::Allow),
            "OON RecommendationsBlacklist should allow after rule removal, got {oon:?}"
        );
    }

    #[test]
    fn abusive_high_recall_allows_follower_on_both_surfaces() {
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();
        let candidate = candidate_with_author_user_label(LabelValue::ABUSIVE_HIGH_RECALL, true);
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        for level in [
            SafetyLevel::TimelineHome,
            SafetyLevel::TimelineHomeRecommendations,
        ] {
            let action = policies.evaluate(level, &viewer, &candidate).action;
            assert!(
                matches!(action, VfAction::Allow),
                "AbusiveHighRecall follower on {level:?} should allow, got {action:?}"
            );
        }
    }

    #[test]
    fn abusive_high_recall_drops_oon_non_follower_but_allows_in_network() {
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();
        let candidate = candidate_with_author_user_label(LabelValue::ABUSIVE_HIGH_RECALL, false);
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network AbusiveHighRecall should allow, got {in_network:?}"
        );

        let oon = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &candidate,
        );
        assert!(
            matches!(oon.action, VfAction::Drop(_)),
            "OON AbusiveHighRecall non-follower should drop, got {:?}",
            oon.action
        );
        assert_eq!(oon.decided_by, Some("AbusiveHighRecallRule"));
    }

    #[test]
    fn nsfw_near_perfect_drops_oon_but_allows_in_network() {
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();
        let candidate = candidate_with_author_user_label(LabelValue::NSFW_NEAR_PERFECT, false);
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let in_network = policies
            .evaluate(SafetyLevel::TimelineHome, &viewer, &candidate)
            .action;
        assert!(
            matches!(in_network, VfAction::Allow),
            "in-network NsfwNearPerfect should allow, got {in_network:?}"
        );

        let oon = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &candidate,
        );
        assert!(
            matches!(oon.action, VfAction::Drop(_)),
            "OON NsfwNearPerfect should drop, got {:?}",
            oon.action
        );
        assert_eq!(oon.decided_by, Some("NsfwNearPerfectAuthorRule"));
    }

    #[test]
    fn do_not_amplify_drops_oon_non_follower_but_allows_follower() {
        use xai_x_thrift::user_labels::LabelValue;
        let policies = Policies::new();
        let viewer = ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        };

        let non_follower = candidate_with_author_user_label(LabelValue::DO_NOT_AMPLIFY, false);
        let oon = policies.evaluate(
            SafetyLevel::TimelineHomeRecommendations,
            &viewer,
            &non_follower,
        );
        assert!(
            matches!(oon.action, VfAction::Drop(_)),
            "OON DoNotAmplify non-follower should drop, got {:?}",
            oon.action
        );
        assert_eq!(oon.decided_by, Some("DoNotAmplifyNonFollowerRule"));

        let follower = candidate_with_author_user_label(LabelValue::DO_NOT_AMPLIFY, true);
        let action = policies
            .evaluate(SafetyLevel::TimelineHomeRecommendations, &viewer, &follower)
            .action;
        assert!(
            matches!(action, VfAction::Allow),
            "OON DoNotAmplify follower should allow, got {action:?}"
        );
    }
}
