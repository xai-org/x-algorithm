use crate::models::{
    AuthorFeatures, ExclusiveContentFeatures, HydratedTweetCandidate, SafetyLabelType,
    TweetFeatures, VfAction, ViewerAge, ViewerAuthorRelationship, ViewerFeatures,
};
use crate::rules::fixtures::{
    author_viewer, candidate, logged_out_viewer, sensitive_opt_in_viewer, viewer, VIEWER_ID,
};
use crate::rules::{RuleEngine, SafetyLevel};
use std::collections::BTreeSet;
use xai_core_entities::entities::{EditControl, EditControlInitial, TakedownReason};
use xai_visibility_filtering::models::{
    Action, DropReason, FilteredReason, SafetyResult, SafetyResultReason,
};
use xai_x_thrift::user_labels::LabelValue;
use SafetyLevel::{FilterAll, TimelineHome, TimelineHomeRecommendations};
use VfAction::{Allow, Drop, Interstitial};

struct Case {
    name: &'static str,
    level: SafetyLevel,
    viewer: ViewerFeatures,
    candidate: HydratedTweetCandidate,
    expected_action: VfAction,
    expected_decided_by: Option<&'static str>,
}

#[test]
fn golden_corpus_pins_policy_verdicts() {
    let rule_engine = RuleEngine::for_tests();
    let mut failures = Vec::new();
    for case in cases() {
        let verdict = rule_engine.evaluate(case.level, &case.viewer, &case.candidate);
        if !action_eq(&verdict.action, &case.expected_action)
            || verdict.decided_by != case.expected_decided_by
        {
            failures.push(format!(
                "{} [{:?}]:\n  expected {:?} decided_by {:?}\n  got      {:?} decided_by {:?}",
                case.name,
                case.level,
                case.expected_action,
                case.expected_decided_by,
                verdict.action,
                verdict.decided_by,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} corpus case(s) diverged:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn every_wired_rule_decides_a_corpus_case() {
    let rule_engine = RuleEngine::for_tests();
    let wired: BTreeSet<&'static str> = [FilterAll, TimelineHome, TimelineHomeRecommendations]
        .into_iter()
        .flat_map(|level| rule_engine.wired_rule_names(level))
        .collect();
    let deciders: BTreeSet<&'static str> = cases()
        .iter()
        .filter_map(|c| c.expected_decided_by)
        .collect();
    let missing: Vec<&&'static str> = wired.difference(&deciders).collect();
    assert!(
        missing.is_empty(),
        "rules wired in RuleEngine but never the decider of any corpus case: {missing:?}"
    );
}

#[test]
fn corpus_case_names_are_unique() {
    let cases = cases();
    let names: BTreeSet<&'static str> = cases.iter().map(|c| c.name).collect();
    assert_eq!(names.len(), cases.len());
}

fn action_eq(left: &VfAction, right: &VfAction) -> bool {
    match (left, right) {
        (Allow, Allow) => true,
        (Drop(l), Drop(r)) => l == r,
        (Interstitial(l), Interstitial(r)) => l == r,
        _ => false,
    }
}

fn cases() -> Vec<Case> {
    let mut cases = filter_all_cases();
    cases.extend(baseline_cases());
    cases.extend(author_state_cases());
    cases.extend(relationship_cases());
    cases.extend(tweet_label_cases());
    cases.extend(tweet_shape_cases());
    cases.extend(age_gating_cases());
    cases.extend(exclusive_content_cases());
    cases.extend(interstitial_cases());
    cases.extend(oon_media_cases());
    cases.extend(oon_tweet_label_cases());
    cases.extend(oon_user_label_cases());
    cases.extend(interaction_cases());
    cases
}

fn author_candidate(set: fn(&mut AuthorFeatures)) -> HydratedTweetCandidate {
    let mut features = AuthorFeatures::default();
    set(&mut features);
    candidate().with_author_features(features).build()
}

fn tweet_candidate(set: fn(&mut TweetFeatures)) -> HydratedTweetCandidate {
    let mut features = TweetFeatures::default();
    set(&mut features);
    candidate().with_tweet_features(features).build()
}

fn relationship_candidate(set: fn(&mut ViewerAuthorRelationship)) -> HydratedTweetCandidate {
    let mut relationship = ViewerAuthorRelationship::default();
    set(&mut relationship);
    candidate().with_relationship(relationship).build()
}

fn labeled(label: SafetyLabelType) -> HydratedTweetCandidate {
    candidate().with_label(label).build()
}

fn labeled_countries(label: SafetyLabelType, countries: &[&str]) -> HydratedTweetCandidate {
    candidate()
        .with_label_countries(
            label,
            countries.iter().map(|c| (*c).to_string()).collect(),
        )
        .build()
}

fn labeled_media(label: SafetyLabelType) -> HydratedTweetCandidate {
    candidate().with_label(label).with_media().build()
}

fn user_labeled(label: LabelValue) -> HydratedTweetCandidate {
    candidate().with_author_user_label(label).build()
}

fn user_labeled_follower(label: LabelValue) -> HydratedTweetCandidate {
    candidate().with_author_user_label(label).followed().build()
}

fn stale_candidate() -> HydratedTweetCandidate {
    tweet_candidate(|t| {
        t.edit_control = Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![1, 2],
            ..Default::default()
        }))
    })
}

fn takedown_candidate(reason: TakedownReason) -> HydratedTweetCandidate {
    candidate()
        .with_tweet_features(TweetFeatures {
            takedown_reasons: vec![reason],
            ..Default::default()
        })
        .build()
}

fn exclusive_candidate(viewer_super_follows_author: bool) -> HydratedTweetCandidate {
    let mut c = candidate().build();
    c.exclusive_content = Some(ExclusiveContentFeatures {
        conversation_author_id: 42,
        viewer_super_follows_author,
    });
    c
}

fn viewer_in_country(code: &str) -> ViewerFeatures {
    ViewerFeatures {
        country_code: Some(code.to_string()),
        ..viewer(VIEWER_ID)
    }
}

fn viewer_with_age(age: ViewerAge) -> ViewerFeatures {
    ViewerFeatures {
        viewer_age: age,
        ..viewer(VIEWER_ID)
    }
}

fn no_stated_age_viewer(account_country_code: &str) -> ViewerFeatures {
    ViewerFeatures {
        account_country_code: Some(account_country_code.to_string()),
        ..viewer_with_age(ViewerAge::NotStated)
    }
}

fn nsfw_high_precision_reason() -> FilteredReason {
    FilteredReason::SafetyResult(SafetyResult {
        reason: Some(SafetyResultReason::NsfwHighPrecision),
        action: Action::Drop(DropReason {}),
    })
}

fn filter_all_cases() -> Vec<Case> {
    vec![
        Case {
            name: "filter_all_drops_pristine_candidate",
            level: FilterAll,
            viewer: viewer(VIEWER_ID),
            candidate: candidate().build(),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("FilterAllRule"),
        },
        Case {
            name: "filter_all_drops_even_self_view",
            level: FilterAll,
            viewer: author_viewer(),
            candidate: candidate().build(),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("FilterAllRule"),
        },
    ]
}

fn baseline_cases() -> Vec<Case> {
    vec![
        Case {
            name: "home_allows_pristine_candidate",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: candidate().build(),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "recommendations_allow_pristine_candidate",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: candidate().build(),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "home_allows_pristine_candidate_for_logged_out",
            level: TimelineHome,
            viewer: logged_out_viewer(),
            candidate: candidate().build(),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "home_allows_egregious_nsfw_tweet_label",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::EGREGIOUS_NSFW),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "recommendations_allow_egregious_nsfw_tweet_label",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::EGREGIOUS_NSFW),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "home_allows_egregious_nsfw_user_label",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::EGREGIOUS_NSFW),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "recommendations_allow_egregious_nsfw_user_label",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::EGREGIOUS_NSFW),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "home_allows_recommendations_blacklist_user_label",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::RECOMMENDATIONS_BLACKLIST),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "recommendations_allow_recommendations_blacklist_user_label",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::RECOMMENDATIONS_BLACKLIST),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn author_state_cases() -> Vec<Case> {
    vec![
        Case {
            name: "suspended_author_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_suspended = true),
            expected_action: Drop(FilteredReason::AuthorIsSuspended),
            expected_decided_by: Some("SuspendedAuthorRule"),
        },
        Case {
            name: "suspended_author_allows_self_view",
            level: TimelineHome,
            viewer: author_viewer(),
            candidate: author_candidate(|a| a.is_suspended = true),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "deactivated_author_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_deactivated = true),
            expected_action: Drop(FilteredReason::AuthorIsDeactivated),
            expected_decided_by: Some("DeactivatedAuthorRule"),
        },
        Case {
            name: "erased_author_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_erased = true),
            expected_action: Drop(FilteredReason::AuthorAccountIsInactive),
            expected_decided_by: Some("ErasedAuthorRule"),
        },
        Case {
            name: "offboarded_author_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_offboarded = true),
            expected_action: Drop(FilteredReason::AuthorAccountIsInactive),
            expected_decided_by: Some("OffboardedAuthorRule"),
        },
        Case {
            name: "protected_author_drops_non_follower",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_protected = true),
            expected_action: Drop(FilteredReason::AuthorIsProtected),
            expected_decided_by: Some("ProtectedAuthorDropRule"),
        },
        Case {
            name: "protected_author_allows_follower",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: {
                let features = AuthorFeatures {
                    is_protected: true,
                    ..Default::default()
                };
                candidate()
                    .with_author_features(features)
                    .followed()
                    .build()
            },
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "protected_author_drops_logged_out",
            level: TimelineHome,
            viewer: logged_out_viewer(),
            candidate: author_candidate(|a| a.is_protected = true),
            expected_action: Drop(FilteredReason::AuthorIsProtected),
            expected_decided_by: Some("ProtectedAuthorDropRule"),
        },
    ]
}

fn relationship_cases() -> Vec<Case> {
    vec![
        Case {
            name: "viewer_blocking_author_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: relationship_candidate(|r| r.viewer_blocks_author = true),
            expected_action: Drop(FilteredReason::ViewerBlocksAuthor),
            expected_decided_by: Some("ViewerBlocksAuthorRule"),
        },
        Case {
            name: "viewer_muting_author_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: relationship_candidate(|r| r.viewer_mutes_author = true),
            expected_action: Drop(FilteredReason::ViewerMutesAuthor),
            expected_decided_by: Some("ViewerMutesAuthorRule"),
        },
        Case {
            name: "block_decides_before_mute",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: relationship_candidate(|r| {
                r.viewer_blocks_author = true;
                r.viewer_mutes_author = true;
            }),
            expected_action: Drop(FilteredReason::ViewerBlocksAuthor),
            expected_decided_by: Some("ViewerBlocksAuthorRule"),
        },
        Case {
            name: "muted_retweets_drop_retweet",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: {
                let relationship = ViewerAuthorRelationship {
                    viewer_mutes_retweets_from_author: true,
                    ..Default::default()
                };
                candidate()
                    .with_relationship(relationship)
                    .retweet_of(2)
                    .build()
            },
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("MutedRetweetsRule"),
        },
        Case {
            name: "muted_retweets_allow_original_tweet",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: relationship_candidate(|r| r.viewer_mutes_retweets_from_author = true),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn tweet_label_cases() -> Vec<Case> {
    vec![
        Case {
            name: "pdna_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::PDNA),
            expected_action: Drop(nsfw_high_precision_reason()),
            expected_decided_by: Some("PdnaTweetLabelRule"),
        },
        Case {
            name: "pdna_label_allows_self_view",
            level: TimelineHome,
            viewer: author_viewer(),
            candidate: labeled(SafetyLabelType::PDNA),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "bounce_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::BOUNCE),
            expected_action: Drop(FilteredReason::TweetIsBounced),
            expected_decided_by: Some("BounceTweetLabelRule"),
        },
        Case {
            name: "spam_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::SPAM),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("SpamTweetLabelRule"),
        },
        Case {
            name: "for_emergency_use_only_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOR_EMERGENCY_USE_ONLY),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("ForEmergencyUseOnlyDropRule"),
        },
        Case {
            name: "for_emergency_use_only_label_drops_even_self_view",
            level: TimelineHome,
            viewer: author_viewer(),
            candidate: labeled(SafetyLabelType::FOR_EMERGENCY_USE_ONLY),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("ForEmergencyUseOnlyDropRule"),
        },
        Case {
            name: "fosnr_hateful_conduct_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOSNR_HATEFUL_CONDUCT),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("FosnrHatefulConductDropRule"),
        },
        Case {
            name: "fosnr_violent_speech_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOSNR_VIOLENT_SPEECH),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("FosnrViolentSpeechDropRule"),
        },
        Case {
            name: "fosnr_abuse_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOSNR_ABUSE),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("FosnrAbuseDropRule"),
        },
        Case {
            name: "fosnr_civic_integrity_label_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOSNR_CIVIC_INTEGRITY),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("FosnrCivicIntegrityDropRule"),
        },
    ]
}

fn tweet_shape_cases() -> Vec<Case> {
    vec![
        Case {
            name: "nullcast_tweet_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: tweet_candidate(|t| t.is_nullcast = true),
            expected_action: Drop(FilteredReason::TweetIsNullcast),
            expected_decided_by: Some("NullcastedTweetDropRule"),
        },
        Case {
            name: "nullcast_retweet_allows",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: {
                let features = TweetFeatures {
                    is_nullcast: true,
                    ..Default::default()
                };
                candidate()
                    .with_tweet_features(features)
                    .retweet_of(2)
                    .build()
            },
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "stale_edit_tweet_drops",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: stale_candidate(),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropStaleTweetsRule"),
        },
        Case {
            name: "legal_takedown_drops_in_withheld_country",
            level: TimelineHome,
            viewer: viewer_in_country("us"),
            candidate: takedown_candidate(TakedownReason::LegalRequest {
                country_code: "us".to_string(),
            }),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropLegalTakendownPostRule"),
        },
        Case {
            name: "legal_takedown_allows_other_country",
            level: TimelineHome,
            viewer: viewer_in_country("fr"),
            candidate: takedown_candidate(TakedownReason::LegalRequest {
                country_code: "us".to_string(),
            }),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "local_laws_takedown_drops_in_withheld_country",
            level: TimelineHome,
            viewer: viewer_in_country("de"),
            candidate: takedown_candidate(TakedownReason::BystanderReport {
                country_code: "de".to_string(),
            }),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropLocalLawsTakendownPostRule"),
        },
        Case {
            name: "legal_takedown_worldwide_drops_for_us_viewer",
            level: TimelineHome,
            viewer: viewer_in_country("us"),
            candidate: takedown_candidate(TakedownReason::LegalRequest {
                country_code: "xx".to_string(),
            }),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropLegalTakendownPostRule"),
        },
        Case {
            name: "legal_takedown_worldwide_drops_without_viewer_country",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: takedown_candidate(TakedownReason::LegalRequest {
                country_code: "xx".to_string(),
            }),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropLegalTakendownPostRule"),
        },
        Case {
            name: "local_laws_takedown_worldwide_drops_for_any_viewer",
            level: TimelineHome,
            viewer: viewer_in_country("us"),
            candidate: takedown_candidate(TakedownReason::BystanderReport {
                country_code: "xx".to_string(),
            }),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropLocalLawsTakendownPostRule"),
        },
        Case {
            name: "dmca_takedown_drops_for_any_viewer",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: takedown_candidate(TakedownReason::Dmca),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropLegalTakendownPostRule"),
        },
        Case {
            name: "dmca_takedown_allows_author",
            level: TimelineHome,
            viewer: author_viewer(),
            candidate: takedown_candidate(TakedownReason::Dmca),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn age_gating_cases() -> Vec<Case> {
    vec![
        Case {
            name: "logged_out_viewer_drops_sensitive_media",
            level: TimelineHome,
            viewer: logged_out_viewer(),
            candidate: labeled_media(SafetyLabelType::NSFW_HIGH_RECALL),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("SensitiveViewerLoggedOutDropRule"),
        },
        Case {
            name: "underage_viewer_drops_sensitive_media",
            level: TimelineHome,
            viewer: viewer_with_age(ViewerAge::Known(17)),
            candidate: labeled_media(SafetyLabelType::NSFW_HIGH_RECALL),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("SensitiveViewerUnderageDropRule"),
        },
        Case {
            name: "no_stated_age_in_gating_country_drops_sensitive_media",
            level: TimelineHome,
            viewer: no_stated_age_viewer("gb"),
            candidate: labeled_media(SafetyLabelType::NSFW_HIGH_RECALL),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("SensitiveViewerNoStatedAgeDropRule"),
        },
        Case {
            name: "no_stated_age_outside_gating_country_allows_sensitive_media",
            level: TimelineHome,
            viewer: no_stated_age_viewer("us"),
            candidate: labeled_media(SafetyLabelType::NSFW_HIGH_RECALL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "known_adult_age_allows_sensitive_media_in_network",
            level: TimelineHome,
            viewer: viewer_with_age(ViewerAge::Known(30)),
            candidate: labeled_media(SafetyLabelType::NSFW_HIGH_RECALL),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn exclusive_content_cases() -> Vec<Case> {
    vec![
        Case {
            name: "exclusive_tweet_drops_non_subscriber",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: exclusive_candidate(false),
            expected_action: Drop(FilteredReason::ExclusiveTweet),
            expected_decided_by: Some("DropExclusiveTweetContentRule"),
        },
        Case {
            name: "exclusive_tweet_allows_super_follower",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: exclusive_candidate(true),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "exclusive_tweet_drops_logged_out",
            level: TimelineHome,
            viewer: logged_out_viewer(),
            candidate: exclusive_candidate(false),
            expected_action: Drop(FilteredReason::ExclusiveTweet),
            expected_decided_by: Some("DropExclusiveTweetContentRule"),
        },
    ]
}

fn interstitial_cases() -> Vec<Case> {
    vec![
        Case {
            name: "nsfw_high_precision_label_interstitials_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::NSFW_HIGH_PRECISION),
            expected_action: Interstitial(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwHighPrecisionInterstitialRule"),
        },
        Case {
            name: "gore_and_violence_label_interstitials_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
            expected_action: Interstitial(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("GoreAndViolenceInterstitialRule"),
        },
        Case {
            name: "nsfw_card_image_label_interstitials_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::NSFW_CARD_IMAGE),
            expected_action: Interstitial(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwCardImageInterstitialRule"),
        },
        Case {
            name: "nsfw_author_with_media_interstitials_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: {
                let features = AuthorFeatures {
                    is_nsfw_user: true,
                    ..Default::default()
                };
                candidate()
                    .with_author_features(features)
                    .with_media()
                    .build()
            },
            expected_action: Interstitial(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwAuthorInterstitialRule"),
        },
        Case {
            name: "nsfw_author_without_media_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_nsfw_user = true),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "nsfw_interstitial_exempts_self_view",
            level: TimelineHome,
            viewer: author_viewer(),
            candidate: labeled(SafetyLabelType::NSFW_HIGH_PRECISION),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "nsfw_interstitial_exempts_sensitive_opt_in_viewer",
            level: TimelineHome,
            viewer: sensitive_opt_in_viewer(),
            candidate: labeled(SafetyLabelType::NSFW_HIGH_PRECISION),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn oon_media_cases() -> Vec<Case> {
    vec![
        Case {
            name: "dmca_media_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: tweet_candidate(|t| t.media.has_dmca_media = true),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropTweetsWithDmcaMediaRule"),
        },
        Case {
            name: "dmca_media_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: tweet_candidate(|t| t.media.has_dmca_media = true),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "geo_denied_media_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer_in_country("de"),
            candidate: tweet_candidate(|t| t.media.geo_deny_list = vec!["de".to_string()]),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropTweetsWithGeoRestrictedMediaRule"),
        },
        Case {
            name: "geo_allow_listed_media_drops_unknown_country_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: tweet_candidate(|t| t.media.geo_allow_list = vec!["us".to_string()]),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DropTweetsWithGeoRestrictedMediaRule"),
        },
        Case {
            name: "nsfw_user_author_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_nsfw_user = true),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("DropNsfwUserAuthorRule"),
        },
        Case {
            name: "nsfw_admin_author_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: author_candidate(|a| a.is_nsfw_admin = true),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("DropNsfwAdminAuthorRule"),
        },
        Case {
            name: "tweet_nsfw_user_flag_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: tweet_candidate(|t| t.nsfw.user = true),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("TweetNsfwUserDropRule"),
        },
        Case {
            name: "tweet_nsfw_user_flag_drops_even_self_view_oon",
            level: TimelineHomeRecommendations,
            viewer: author_viewer(),
            candidate: tweet_candidate(|t| t.nsfw.user = true),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("TweetNsfwUserDropRule"),
        },
        Case {
            name: "tweet_nsfw_admin_flag_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: tweet_candidate(|t| t.nsfw.admin = true),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("TweetNsfwAdminDropRule"),
        },
    ]
}

fn oon_tweet_label_cases() -> Vec<Case> {
    vec![
        Case {
            name: "nsfw_high_recall_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::NSFW_HIGH_RECALL),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwHighRecallDropRule"),
        },
        Case {
            name: "nsfw_high_precision_label_drop_beats_interstitial_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::NSFW_HIGH_PRECISION),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwHighPrecisionOonDropRule"),
        },
        Case {
            name: "gore_and_violence_label_drop_beats_interstitial_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("GoreAndViolenceOonDropRule"),
        },
        Case {
            name: "nsfw_card_image_label_drop_beats_interstitial_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::NSFW_CARD_IMAGE),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwCardImageOonDropRule"),
        },
        Case {
            name: "sensitive_opt_in_does_not_save_oon_drop",
            level: TimelineHomeRecommendations,
            viewer: sensitive_opt_in_viewer(),
            candidate: labeled(SafetyLabelType::NSFW_HIGH_PRECISION),
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("NsfwHighPrecisionOonDropRule"),
        },
        Case {
            name: "do_not_amplify_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::DO_NOT_AMPLIFY),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("DoNotAmplifyOonDropRule"),
        },
        Case {
            name: "country_scoped_do_not_amplify_allows_out_of_country_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer_in_country("us"),
            candidate: labeled_countries(SafetyLabelType::DO_NOT_AMPLIFY, &["br"]),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "country_scoped_do_not_amplify_drops_in_country_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer_in_country("br"),
            candidate: labeled_countries(SafetyLabelType::DO_NOT_AMPLIFY, &["br"]),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("DoNotAmplifyOonDropRule"),
        },
        Case {
            name: "malicious_url_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::MALICIOUS_URL),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("MaliciousUrlOonDropRule"),
        },
        Case {
            name: "malicious_url_label_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::MALICIOUS_URL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "malicious_url_label_allows_self_view_oon",
            level: TimelineHomeRecommendations,
            viewer: author_viewer(),
            candidate: labeled(SafetyLabelType::MALICIOUS_URL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "spam_high_recall_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::SPAM_HIGH_RECALL),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("SpamHighRecallDropRule"),
        },
        Case {
            name: "spam_high_recall_label_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::SPAM_HIGH_RECALL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "nsfw_text_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::NSFW_TEXT),
            expected_action: Drop(nsfw_high_precision_reason()),
            expected_decided_by: Some("NsfwTextTweetLabelDropRule"),
        },
        Case {
            name: "fosnr_abuse_insults_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOSNR_ABUSE_INSULTS),
            expected_action: Drop(FilteredReason::PossiblyUndesirable),
            expected_decided_by: Some("FosnrAbuseInsultsOonDropRule"),
        },
        Case {
            name: "fosnr_abuse_insults_label_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: labeled(SafetyLabelType::FOSNR_ABUSE_INSULTS),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn oon_user_label_cases() -> Vec<Case> {
    vec![
        Case {
            name: "nsfw_high_recall_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_HIGH_RECALL),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("NsfwHighRecallUserLabelRule"),
        },
        Case {
            name: "nsfw_high_recall_user_label_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_HIGH_RECALL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "nsfw_high_recall_user_label_allows_self_view_oon",
            level: TimelineHomeRecommendations,
            viewer: author_viewer(),
            candidate: user_labeled(LabelValue::NSFW_HIGH_RECALL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "nsfw_high_precision_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_HIGH_PRECISION),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("NsfwHighPrecisionUserLabelRule"),
        },
        Case {
            name: "spam_high_recall_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::SPAM_HIGH_RECALL),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("SpamHighRecallUserLabelRule"),
        },
        Case {
            name: "compromised_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::COMPROMISED),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("CompromisedUserLabelRule"),
        },
        Case {
            name: "read_only_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::READ_ONLY),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("ReadOnlyUserLabelRule"),
        },
        Case {
            name: "impersonation_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::IMPERSONATION_HIGH_PRECISION),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("ImpersonationHighPrecisionUserLabelRule"),
        },
        Case {
            name: "nsfw_avatar_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_AVATAR_IMAGE),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("NsfwAvatarImageRule"),
        },
        Case {
            name: "nsfw_banner_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_BANNER_IMAGE),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("NsfwBannerImageRule"),
        },
        Case {
            name: "abusive_high_recall_user_label_drops_non_follower_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::ABUSIVE_HIGH_RECALL),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("AbusiveHighRecallRule"),
        },
        Case {
            name: "abusive_high_recall_user_label_allows_follower_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled_follower(LabelValue::ABUSIVE_HIGH_RECALL),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "nsfw_near_perfect_user_label_drops_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_NEAR_PERFECT),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("NsfwNearPerfectAuthorRule"),
        },
        Case {
            name: "nsfw_near_perfect_user_label_allows_in_network",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::NSFW_NEAR_PERFECT),
            expected_action: Allow,
            expected_decided_by: None,
        },
        Case {
            name: "do_not_amplify_user_label_drops_non_follower_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled(LabelValue::DO_NOT_AMPLIFY),
            expected_action: Drop(FilteredReason::UnspecifiedReason),
            expected_decided_by: Some("DoNotAmplifyNonFollowerRule"),
        },
        Case {
            name: "do_not_amplify_user_label_allows_follower_oon",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: user_labeled_follower(LabelValue::DO_NOT_AMPLIFY),
            expected_action: Allow,
            expected_decided_by: None,
        },
    ]
}

fn interaction_cases() -> Vec<Case> {
    vec![
        Case {
            name: "drop_short_circuits_before_interstitial_attribution",
            level: TimelineHome,
            viewer: viewer(VIEWER_ID),
            candidate: {
                let features = AuthorFeatures {
                    is_suspended: true,
                    ..Default::default()
                };
                candidate()
                    .with_author_features(features)
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_media()
                    .build()
            },
            expected_action: Drop(FilteredReason::AuthorIsSuspended),
            expected_decided_by: Some("SuspendedAuthorRule"),
        },
        Case {
            name: "later_oon_drop_beats_earlier_nsfw_author_interstitial",
            level: TimelineHomeRecommendations,
            viewer: viewer(VIEWER_ID),
            candidate: {
                let features = AuthorFeatures {
                    is_nsfw_user: true,
                    ..Default::default()
                };
                candidate()
                    .with_author_features(features)
                    .with_media()
                    .build()
            },
            expected_action: Drop(FilteredReason::ContainNsfwMedia),
            expected_decided_by: Some("DropNsfwUserAuthorRule"),
        },
    ]
}
