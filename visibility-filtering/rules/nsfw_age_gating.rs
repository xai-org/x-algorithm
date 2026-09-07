use crate::models::{SafetyLabelType, VfAction, ViewerFeatures};
use crate::rules::{Rule, RuleContext};
use xai_visibility_filtering::models::FilteredReason;

const NSFW_GATING_COUNTRIES: [&str; 16] = [
    "ar", "au", "br", "ca", "de", "es", "fr", "gb", "id", "it", "kr", "mx", "nl", "ph", "pt", "th",
];

fn nsfw_base_condition(context: &RuleContext<'_>) -> bool {
    let candidate = context.candidate();
    !context.is_author_viewer()
        && candidate.has_media()
        && (context.has_tweet_safety_label(SafetyLabelType::NSFW_HIGH_PRECISION)
            || context.has_tweet_safety_label(SafetyLabelType::NSFW_HIGH_RECALL)
            || (candidate.is_nsfw_flagged() && !candidate.is_retweet()))
}

fn graphic_base_condition(context: &RuleContext<'_>) -> bool {
    let candidate = context.candidate();
    !context.is_author_viewer()
        && candidate.has_media()
        && context.has_tweet_safety_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)
}

fn nsfw_no_media_label_condition(context: &RuleContext<'_>) -> bool {
    !context.is_author_viewer()
        && (context.has_tweet_safety_label(SafetyLabelType::NSFW_TEXT)
            || context.has_tweet_safety_label(SafetyLabelType::NSFW_CARD_IMAGE))
}

fn sensitive_base_condition(context: &RuleContext<'_>) -> bool {
    nsfw_base_condition(context)
        || graphic_base_condition(context)
        || nsfw_no_media_label_condition(context)
}

fn viewer_in_gating_jurisdiction(viewer: &ViewerFeatures) -> bool {
    viewer
        .account_country_code
        .as_deref()
        .or(viewer.country_code.as_deref())
        .is_some_and(|c| NSFW_GATING_COUNTRIES.contains(&c))
}

pub struct SensitiveViewerLoggedOutDropRule;

impl Rule for SensitiveViewerLoggedOutDropRule {
    fn name(&self) -> &'static str {
        "SensitiveViewerLoggedOutDropRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        if viewer.viewer_is_logged_out() && sensitive_base_condition(context) {
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        } else {
            VfAction::Allow
        }
    }
}

pub struct SensitiveViewerUnderageDropRule;

impl Rule for SensitiveViewerUnderageDropRule {
    fn name(&self) -> &'static str {
        "SensitiveViewerUnderageDropRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        if viewer.viewer_is_underage() && sensitive_base_condition(context) {
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        } else {
            VfAction::Allow
        }
    }
}

pub struct SensitiveViewerNoStatedAgeDropRule;

impl Rule for SensitiveViewerNoStatedAgeDropRule {
    fn name(&self) -> &'static str {
        "SensitiveViewerNoStatedAgeDropRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        if viewer.viewer_has_no_stated_age()
            && viewer_in_gating_jurisdiction(viewer)
            && sensitive_base_condition(context)
        {
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        } else {
            VfAction::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        AuthorFeatures, HydratedTweetCandidate, MediaFeature, NsfwFeature, SafetyLabel,
        SafetyLabelMap, TweetFeatures, Viewer, ViewerAge, ViewerFeatures,
    };
    use std::collections::HashMap;

    fn viewer(age: ViewerAge) -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            allows_sensitive_media: false,
            viewer_age: age,
            country_code: Some("de".into()),
            account_country_code: None,
        }
    }

    fn media_candidate_with_label(label: SafetyLabelType) -> HydratedTweetCandidate {
        let mut labels = HashMap::new();
        labels.insert(label, SafetyLabel::default());
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            tweet_features: TweetFeatures {
                media: MediaFeature {
                    has_media: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn nsfw_author_media_candidate() -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            tweet_features: TweetFeatures {
                media: MediaFeature {
                    has_media: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            author_features: AuthorFeatures {
                is_nsfw_user: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn underage_drops_nsfw_label_media() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_nsfw_high_recall_media() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_RECALL);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_gore_label_media() {
        let c = media_candidate_with_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drop_even_when_opted_in() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            allows_sensitive_media: true,
            ..viewer(ViewerAge::Known(15))
        };
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn adult_does_not_drop() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(18)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn unknown_age_is_not_confirmed_underage() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerUnderageDropRule
                .evaluate(&crate::rules::test_context(&viewer(ViewerAge::Unknown), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn unknown_age_drops_like_no_stated_age_in_gating_jurisdiction() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule
                .evaluate(&crate::rules::test_context(&viewer(ViewerAge::Unknown), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn unknown_age_allows_outside_gating_jurisdiction() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            country_code: Some("us".into()),
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn logged_out_unknown_is_not_handled_by_no_stated_age_rule() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    fn no_media_candidate_with_label(label: SafetyLabelType) -> HydratedTweetCandidate {
        let mut candidate = media_candidate_with_label(label);
        candidate.tweet_features.media.has_media = false;
        candidate
    }

    #[test]
    fn underage_drops_nsfw_text_without_media() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_nsfw_card_image_without_media() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_CARD_IMAGE);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn no_stated_age_drops_nsfw_text_in_jurisdiction() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::NotStated),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn no_stated_age_allows_nsfw_text_outside_jurisdiction() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        let v = ViewerFeatures {
            country_code: Some("us".into()),
            ..viewer(ViewerAge::NotStated)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn logged_out_drops_nsfw_text_without_media() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        let v = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerLoggedOutDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn logged_out_drops_nsfw_card_image_without_media() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_CARD_IMAGE);
        let v = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerLoggedOutDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn adult_allows_nsfw_text() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(18)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn unknown_age_is_not_confirmed_underage_for_nsfw_text() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        assert!(matches!(
            SensitiveViewerUnderageDropRule
                .evaluate(&crate::rules::test_context(&viewer(ViewerAge::Unknown), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn unknown_age_drops_nsfw_text_in_gating_jurisdiction() {
        let c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule
                .evaluate(&crate::rules::test_context(&viewer(ViewerAge::Unknown), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn nsfw_text_self_view_is_exempt() {
        let mut c = no_media_candidate_with_label(SafetyLabelType::NSFW_TEXT);
        c.author_id = 999;
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn label_rule_requires_media() {
        let mut c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        c.tweet_features.media.has_media = false;
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn self_view_is_exempt() {
        let mut c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        c.author_id = 999;
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn no_stated_age_drops_in_jurisdiction() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::NotStated),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn no_stated_age_allows_outside_jurisdiction() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            country_code: Some("us".into()),
            ..viewer(ViewerAge::NotStated)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn no_stated_age_allows_missing_country() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            country_code: None,
            ..viewer(ViewerAge::NotStated)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn no_stated_age_allows_non_gating_account_country_over_gating_request() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            country_code: Some("de".into()),
            account_country_code: Some("us".into()),
            ..viewer(ViewerAge::NotStated)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn no_stated_age_drops_gating_account_country() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            country_code: Some("us".into()),
            account_country_code: Some("kr".into()),
            ..viewer(ViewerAge::NotStated)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn no_stated_age_falls_back_to_request_country_when_account_absent() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            country_code: Some("de".into()),
            account_country_code: None,
            ..viewer(ViewerAge::NotStated)
        };
        assert!(matches!(
            SensitiveViewerNoStatedAgeDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_nsfw_author_media() {
        let c = nsfw_author_media_candidate();
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_nsfw_admin_author_media() {
        let mut c = nsfw_author_media_candidate();
        c.author_features = AuthorFeatures {
            is_nsfw_user: false,
            is_nsfw_admin: true,
            ..Default::default()
        };
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    fn nsfw_tweet_flag_media_candidate() -> HydratedTweetCandidate {
        let mut c = nsfw_author_media_candidate();
        c.author_features = AuthorFeatures::default();
        c.tweet_features.nsfw = NsfwFeature {
            user: true,
            admin: false,
        };
        c
    }

    #[test]
    fn underage_drops_nsfw_tweet_flag_media() {
        let c = nsfw_tweet_flag_media_candidate();
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_nsfw_admin_tweet_flag_media() {
        let mut c = nsfw_tweet_flag_media_candidate();
        c.tweet_features.nsfw = NsfwFeature {
            user: false,
            admin: true,
        };
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_drops_when_both_flag_sources_set() {
        let mut c = nsfw_tweet_flag_media_candidate();
        c.author_features.is_nsfw_user = true;
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn underage_allows_no_flags_no_labels() {
        let mut c = nsfw_tweet_flag_media_candidate();
        c.tweet_features.nsfw = NsfwFeature::default();
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn nsfw_tweet_flag_retweet_not_dropped() {
        let mut c = nsfw_tweet_flag_media_candidate();
        c.tweet_features.core.source_tweet_id = Some(42);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn nsfw_tweet_flag_self_view_exempt() {
        let mut c = nsfw_tweet_flag_media_candidate();
        c.author_id = 999;
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn nsfw_author_retweet_not_dropped() {
        let mut c = nsfw_author_media_candidate();
        c.tweet_features.core.source_tweet_id = Some(42);
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn nsfw_author_without_media_not_dropped() {
        let mut c = nsfw_author_media_candidate();
        c.tweet_features.media.has_media = false;
        assert!(matches!(
            SensitiveViewerUnderageDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn logged_out_drops_nsfw_label_media() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let v = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerLoggedOutDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn logged_out_drops_nsfw_author_media() {
        let c = nsfw_author_media_candidate();
        let v = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerLoggedOutDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn logged_out_requires_media() {
        let mut c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        c.tweet_features.media.has_media = false;
        let v = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..viewer(ViewerAge::Unknown)
        };
        assert!(matches!(
            SensitiveViewerLoggedOutDropRule.evaluate(&crate::rules::test_context(&v, &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn logged_in_not_handled_by_logged_out_rule() {
        let c = media_candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            SensitiveViewerLoggedOutDropRule.evaluate(&crate::rules::test_context(
                &viewer(ViewerAge::Known(15)),
                &c
            )),
            VfAction::Allow
        ));
    }
}
