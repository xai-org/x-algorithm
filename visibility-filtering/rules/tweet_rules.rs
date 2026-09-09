use crate::models::{SafetyLabelType, VfAction};
use crate::rules::rule_spec::{RuleAction, RuleSpec};
use crate::rules::RuleContext;
use xai_visibility_filtering::models::{
    Action, DropReason, FilteredReason, SafetyResult, SafetyResultReason,
};

const NSFW_HIGH_PRECISION_REASON: FilteredReason = FilteredReason::SafetyResult(SafetyResult {
    reason: Some(SafetyResultReason::NsfwHighPrecision),
    action: Action::Drop(DropReason {}),
});

pub(super) const TWEET_LABEL_DROPS: &[RuleSpec] = &[
    RuleSpec::Tweet {
        name: "PdnaTweetLabelRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::PDNA),
        action: RuleAction::Drop(NSFW_HIGH_PRECISION_REASON),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "BounceTweetLabelRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::BOUNCE),
        action: RuleAction::Drop(FilteredReason::TweetIsBounced),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "SpamTweetLabelRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::SPAM),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "ForEmergencyUseOnlyDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::FOR_EMERGENCY_USE_ONLY),
        action: RuleAction::Drop(FilteredReason::UnspecifiedReason),
        exempt_author: false,
    },
    RuleSpec::Tweet {
        name: "FosnrHatefulConductDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::FOSNR_HATEFUL_CONDUCT),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "FosnrViolentSpeechDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::FOSNR_VIOLENT_SPEECH),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "FosnrAbuseDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::FOSNR_ABUSE),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "FosnrCivicIntegrityDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::FOSNR_CIVIC_INTEGRITY),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
];

pub(super) const NSFW_MEDIA_INTERSTITIALS: &[RuleSpec] = &[
    RuleSpec::Tweet {
        name: "NsfwHighPrecisionInterstitialRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::NSFW_HIGH_PRECISION),
        action: RuleAction::SensitiveMediaInterstitial(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "GoreAndViolenceInterstitialRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        action: RuleAction::SensitiveMediaInterstitial(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "NsfwCardImageInterstitialRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::NSFW_CARD_IMAGE),
        action: RuleAction::SensitiveMediaInterstitial(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
];

pub(super) const OON_TWEET_FLAG_DROPS: &[RuleSpec] = &[
    RuleSpec::Tweet {
        name: "TweetNsfwUserDropRule",
        when: |tweet| tweet.has_nsfw_user_flag(),
        action: RuleAction::Drop(FilteredReason::ContainNsfwMedia),
        exempt_author: false,
    },
    RuleSpec::Tweet {
        name: "TweetNsfwAdminDropRule",
        when: |tweet| tweet.has_nsfw_admin_flag(),
        action: RuleAction::Drop(FilteredReason::ContainNsfwMedia),
        exempt_author: false,
    },
];

pub(super) const OON_TWEET_LABEL_DROPS: &[RuleSpec] = &[
    RuleSpec::Tweet {
        name: "NsfwHighRecallDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::NSFW_HIGH_RECALL),
        action: RuleAction::Drop(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "NsfwHighPrecisionOonDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::NSFW_HIGH_PRECISION),
        action: RuleAction::Drop(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "GoreAndViolenceOonDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
        action: RuleAction::Drop(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "NsfwCardImageOonDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::NSFW_CARD_IMAGE),
        action: RuleAction::Drop(FilteredReason::ContainNsfwMedia),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "DoNotAmplifyOonDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::DO_NOT_AMPLIFY),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "MaliciousUrlOonDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::MALICIOUS_URL),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "SpamHighRecallDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::SPAM_HIGH_RECALL),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "NsfwTextTweetLabelDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::NSFW_TEXT),
        action: RuleAction::Drop(NSFW_HIGH_PRECISION_REASON),
        exempt_author: true,
    },
    RuleSpec::Tweet {
        name: "FosnrAbuseInsultsOonDropRule",
        when: |tweet| tweet.has_safety_label(SafetyLabelType::FOSNR_ABUSE_INSULTS),
        action: RuleAction::Drop(FilteredReason::PossiblyUndesirable),
        exempt_author: true,
    },
];

fn drop_exclusive_tweet_content(context: &RuleContext<'_>) -> VfAction {
    if !context.tweet().is_exclusive() {
        return VfAction::Allow;
    }

    if context.viewer().is_logged_out() {
        return VfAction::Drop(FilteredReason::ExclusiveTweet);
    }

    if context.viewer().is_conversation_author() {
        return VfAction::Allow;
    }

    if context.viewer().super_follows_author() {
        return VfAction::Allow;
    }

    if !context.tweet().is_retweet() && context.viewer().is_author() {
        return VfAction::Allow;
    }

    VfAction::Drop(FilteredReason::ExclusiveTweet)
}

pub(super) const EXCLUSIVE_TWEET_DROP: &[RuleSpec] = &[RuleSpec::Custom {
    name: "DropExclusiveTweetContentRule",
    evaluate: drop_exclusive_tweet_content,
}];

pub(super) const NSFW_AUTHOR_INTERSTITIAL: &[RuleSpec] = &[RuleSpec::Tweet {
    name: "NsfwAuthorInterstitialRule",
    when: |tweet| tweet.is_nsfw_flagged() && tweet.has_media(),
    action: RuleAction::SensitiveMediaInterstitial(FilteredReason::ContainNsfwMedia),
    exempt_author: true,
}];

fn nsfw_base_condition(context: &RuleContext<'_>) -> bool {
    let tweet = context.tweet();
    !context.viewer().is_author()
        && tweet.has_media()
        && (tweet.has_safety_label(SafetyLabelType::NSFW_HIGH_PRECISION)
            || tweet.has_safety_label(SafetyLabelType::NSFW_HIGH_RECALL)
            || (tweet.is_nsfw_flagged() && !tweet.is_retweet()))
}

fn graphic_base_condition(context: &RuleContext<'_>) -> bool {
    let tweet = context.tweet();
    !context.viewer().is_author()
        && tweet.has_media()
        && tweet.has_safety_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)
}

fn nsfw_no_media_label_condition(context: &RuleContext<'_>) -> bool {
    let tweet = context.tweet();
    !context.viewer().is_author()
        && (tweet.has_safety_label(SafetyLabelType::NSFW_TEXT)
            || tweet.has_safety_label(SafetyLabelType::NSFW_CARD_IMAGE))
}

fn sensitive_base_condition(context: &RuleContext<'_>) -> bool {
    nsfw_base_condition(context)
        || graphic_base_condition(context)
        || nsfw_no_media_label_condition(context)
}

fn sensitive_viewer_logged_out(context: &RuleContext<'_>) -> VfAction {
    if context.viewer().is_logged_out() && sensitive_base_condition(context) {
        VfAction::Drop(FilteredReason::ContainNsfwMedia)
    } else {
        VfAction::Allow
    }
}

fn sensitive_viewer_underage(context: &RuleContext<'_>) -> VfAction {
    if context.viewer().is_underage() && sensitive_base_condition(context) {
        VfAction::Drop(FilteredReason::ContainNsfwMedia)
    } else {
        VfAction::Allow
    }
}

fn sensitive_viewer_no_stated_age(context: &RuleContext<'_>) -> VfAction {
    if context.viewer().has_no_stated_age()
        && context
            .viewer()
            .country()
            .is_some_and(|country| context.nsfw_gating_country(country))
        && sensitive_base_condition(context)
    {
        VfAction::Drop(FilteredReason::ContainNsfwMedia)
    } else {
        VfAction::Allow
    }
}

pub(super) const NULLCAST_DROP: &[RuleSpec] = &[RuleSpec::Tweet {
    name: "NullcastedTweetDropRule",
    when: |tweet| tweet.is_nullcast() && !tweet.is_retweet() && !tweet.is_community_tweet(),
    action: RuleAction::Drop(FilteredReason::TweetIsNullcast),
    exempt_author: false,
}];

fn drop_legal_takendown_post(context: &RuleContext<'_>) -> VfAction {
    if !context.viewer().is_author() && context.takedown().legal_in_viewer_country() {
        return VfAction::Drop(FilteredReason::UnspecifiedReason);
    }
    VfAction::Allow
}

fn drop_local_laws_takendown_post(context: &RuleContext<'_>) -> VfAction {
    if !context.viewer().is_author() && context.takedown().local_laws_in_viewer_country() {
        return VfAction::Drop(FilteredReason::UnspecifiedReason);
    }
    VfAction::Allow
}

fn drop_geo_restricted_media(context: &RuleContext<'_>) -> VfAction {
    if context.takedown().media_restricted_in_viewer_country() {
        VfAction::Drop(FilteredReason::UnspecifiedReason)
    } else {
        VfAction::Allow
    }
}

pub(super) const SAFETY_HYDRATION_FAILURE_DROP: &[RuleSpec] = &[RuleSpec::Custom {
    name: "SafetyHydrationFailureDropRule",
    evaluate: drop_failed_safety_hydration,
}];

fn drop_failed_safety_hydration(context: &RuleContext<'_>) -> VfAction {
    if context.tweet().safety_hydration_failed() {
        VfAction::Drop(FilteredReason::UnspecifiedReason)
    } else {
        VfAction::Allow
    }
}

pub(super) const TES_HOME_DROPS: &[RuleSpec] = &[
    RuleSpec::Tweet {
        name: "DropStaleTweetsRule",
        when: |tweet| tweet.is_stale() && !tweet.is_retweet(),
        action: RuleAction::Drop(FilteredReason::UnspecifiedReason),
        exempt_author: false,
    },
    RuleSpec::Custom {
        name: "DropLegalTakendownPostRule",
        evaluate: drop_legal_takendown_post,
    },
    RuleSpec::Custom {
        name: "DropLocalLawsTakendownPostRule",
        evaluate: drop_local_laws_takendown_post,
    },
];

pub(super) const FILTER_ALL: &[RuleSpec] = &[RuleSpec::Tweet {
    name: "FilterAllRule",
    when: |_| true,
    action: RuleAction::Drop(FilteredReason::UnspecifiedReason),
    exempt_author: false,
}];

pub(super) const RECS_MEDIA_DROPS: &[RuleSpec] = &[
    RuleSpec::Tweet {
        name: "DropTweetsWithDmcaMediaRule",
        when: |tweet| tweet.has_dmca_media(),
        action: RuleAction::Drop(FilteredReason::UnspecifiedReason),
        exempt_author: false,
    },
    RuleSpec::Custom {
        name: "DropTweetsWithGeoRestrictedMediaRule",
        evaluate: drop_geo_restricted_media,
    },
];

pub(super) const SENSITIVE_VIEWER_DROPS: &[RuleSpec] = &[
    RuleSpec::Custom {
        name: "SensitiveViewerLoggedOutDropRule",
        evaluate: sensitive_viewer_logged_out,
    },
    RuleSpec::Custom {
        name: "SensitiveViewerUnderageDropRule",
        evaluate: sensitive_viewer_underage,
    },
    RuleSpec::Custom {
        name: "SensitiveViewerNoStatedAgeDropRule",
        evaluate: sensitive_viewer_no_stated_age,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        AuthorFeatures, ExclusiveContentFeatures, HydratedTweetCandidate, MediaFeature,
        NsfwFeature, TweetFeatures, VfAction, Viewer, ViewerAge, ViewerFeatures,
    };
    use crate::rules::fixtures::{
        assert_allows, assert_drops, author_viewer, candidate, logged_out_viewer,
        nsfw_flag_media_candidates, sensitive_opt_in_viewer, viewer, VIEWER_ID,
    };
    use crate::rules::test_context;
    use xai_core_entities::entities::{EditControl, EditControlInitial, TakedownReason};

    fn trigger_label(name: &str) -> SafetyLabelType {
        match name {
            "PdnaTweetLabelRule" => SafetyLabelType::PDNA,
            "BounceTweetLabelRule" => SafetyLabelType::BOUNCE,
            "SpamTweetLabelRule" => SafetyLabelType::SPAM,
            "ForEmergencyUseOnlyDropRule" => SafetyLabelType::FOR_EMERGENCY_USE_ONLY,
            "FosnrHatefulConductDropRule" => SafetyLabelType::FOSNR_HATEFUL_CONDUCT,
            "FosnrViolentSpeechDropRule" => SafetyLabelType::FOSNR_VIOLENT_SPEECH,
            "FosnrAbuseDropRule" => SafetyLabelType::FOSNR_ABUSE,
            "FosnrCivicIntegrityDropRule" => SafetyLabelType::FOSNR_CIVIC_INTEGRITY,
            "NsfwHighRecallDropRule" => SafetyLabelType::NSFW_HIGH_RECALL,
            "NsfwHighPrecisionOonDropRule" => SafetyLabelType::NSFW_HIGH_PRECISION,
            "GoreAndViolenceOonDropRule" => SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION,
            "NsfwCardImageOonDropRule" => SafetyLabelType::NSFW_CARD_IMAGE,
            "DoNotAmplifyOonDropRule" => SafetyLabelType::DO_NOT_AMPLIFY,
            "MaliciousUrlOonDropRule" => SafetyLabelType::MALICIOUS_URL,
            "SpamHighRecallDropRule" => SafetyLabelType::SPAM_HIGH_RECALL,
            "NsfwTextTweetLabelDropRule" => SafetyLabelType::NSFW_TEXT,
            "FosnrAbuseInsultsOonDropRule" => SafetyLabelType::FOSNR_ABUSE_INSULTS,
            "NsfwHighPrecisionInterstitialRule" => SafetyLabelType::NSFW_HIGH_PRECISION,
            "GoreAndViolenceInterstitialRule" => SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION,
            "NsfwCardImageInterstitialRule" => SafetyLabelType::NSFW_CARD_IMAGE,
            _ => panic!("no trigger label for rule {name}"),
        }
    }

    const UNRELATED_LABEL: SafetyLabelType = SafetyLabelType::EGREGIOUS_NSFW;

    #[test]
    fn tweet_label_drop_axis() {
        for spec in TWEET_LABEL_DROPS.iter().chain(OON_TWEET_LABEL_DROPS) {
            let RuleSpec::Tweet {
                name,
                action: RuleAction::Drop(reason),
                exempt_author,
                ..
            } = spec
            else {
                panic!("{} is not a tweet-label drop row", spec.name());
            };
            let firing = candidate().with_label(trigger_label(name)).build();
            for v in [
                viewer(VIEWER_ID),
                logged_out_viewer(),
                sensitive_opt_in_viewer(),
            ] {
                assert_drops(spec, &v, &firing, reason);
            }
            let followed = candidate()
                .with_label(trigger_label(name))
                .followed()
                .build();
            assert_drops(spec, &viewer(VIEWER_ID), &followed, reason);

            let unrelated = candidate().with_label(UNRELATED_LABEL).build();
            assert_allows(spec, &viewer(VIEWER_ID), &unrelated);

            if *exempt_author {
                assert_allows(spec, &author_viewer(), &firing);
            } else {
                assert_drops(spec, &author_viewer(), &firing, reason);
            }
        }
    }

    #[test]
    fn nsfw_media_interstitial_axis() {
        for spec in NSFW_MEDIA_INTERSTITIALS {
            let RuleSpec::Tweet {
                name,
                action: RuleAction::SensitiveMediaInterstitial(reason),
                exempt_author: true,
                ..
            } = spec
            else {
                panic!(
                    "{} is not an author-exempt sensitive-media interstitial row",
                    spec.name()
                );
            };
            let firing = candidate().with_label(trigger_label(name)).build();
            let action = spec.evaluate(&test_context(&viewer(VIEWER_ID), &firing));
            assert!(
                matches!(&action, VfAction::Interstitial(r) if r == reason),
                "{name} should interstitial, got {action:?}"
            );
            assert_allows(spec, &sensitive_opt_in_viewer(), &firing);
            assert_allows(spec, &author_viewer(), &firing);
            let unrelated = candidate().with_label(UNRELATED_LABEL).build();
            assert_allows(spec, &viewer(VIEWER_ID), &unrelated);
        }
    }

    fn tweet_flag_features(name: &str) -> NsfwFeature {
        match name {
            "TweetNsfwUserDropRule" => NsfwFeature {
                user: true,
                admin: false,
            },
            "TweetNsfwAdminDropRule" => NsfwFeature {
                user: false,
                admin: true,
            },
            _ => panic!("no trigger flags for rule {name}"),
        }
    }

    #[test]
    fn tweet_flag_drop_axis() {
        for spec in OON_TWEET_FLAG_DROPS {
            let RuleSpec::Tweet {
                name,
                action: RuleAction::Drop(reason),
                exempt_author,
                ..
            } = spec
            else {
                panic!("{} is not a tweet-flag drop row", spec.name());
            };
            let nsfw = tweet_flag_features(name);
            let firing = candidate()
                .with_tweet_features(TweetFeatures {
                    nsfw,
                    ..Default::default()
                })
                .build();
            assert_drops(spec, &viewer(VIEWER_ID), &firing, reason);
            let unflagged = candidate().build();
            assert_allows(spec, &viewer(VIEWER_ID), &unflagged);
            if *exempt_author {
                assert_allows(spec, &author_viewer(), &firing);
            } else {
                assert_drops(spec, &author_viewer(), &firing, reason);
            }
        }
    }

    fn exclusive_candidate(
        tweet_id: u64,
        author_id: u64,
        root_author_id: u64,
    ) -> HydratedTweetCandidate {
        let mut c = candidate().tweet_id(tweet_id).author_id(author_id).build();
        c.exclusive_content = Some(ExclusiveContentFeatures {
            conversation_author_id: root_author_id,
            viewer_super_follows_author: false,
        });
        c
    }

    #[test]
    fn nsfw_author_interstitial_axis() {
        let spec = &NSFW_AUTHOR_INTERSTITIAL[0];
        let RuleSpec::Tweet {
            name: "NsfwAuthorInterstitialRule",
            action: RuleAction::SensitiveMediaInterstitial(reason),
            exempt_author: true,
            ..
        } = spec
        else {
            panic!("{} is not the NSFW-author interstitial row", spec.name());
        };
        for firing in nsfw_flag_media_candidates() {
            let action = spec.evaluate(&test_context(&viewer(VIEWER_ID), &firing));
            assert!(
                matches!(&action, VfAction::Interstitial(r) if r == reason),
                "{} should interstitial, got {action:?}",
                spec.name()
            );
            assert_allows(spec, &sensitive_opt_in_viewer(), &firing);
            assert_allows(spec, &author_viewer(), &firing);
        }
        let [mut no_media, ..] = nsfw_flag_media_candidates();
        no_media.tweet_features.media.has_media = false;
        assert_allows(spec, &viewer(VIEWER_ID), &no_media);
        let [_, mut no_flags, ..] = nsfw_flag_media_candidates();
        no_flags.tweet_features.nsfw = NsfwFeature::default();
        assert_allows(spec, &viewer(VIEWER_ID), &no_flags);
    }

    #[test]
    fn exclusive_content_axis() {
        let spec = &EXCLUSIVE_TWEET_DROP[0];
        assert_allows(spec, &viewer(VIEWER_ID), &candidate().build());

        let exclusive = exclusive_candidate(1, 100, 100);
        assert_drops(
            spec,
            &logged_out_viewer(),
            &exclusive,
            &FilteredReason::ExclusiveTweet,
        );
        assert_allows(spec, &author_viewer(), &exclusive);
        assert_drops(
            spec,
            &viewer(200),
            &exclusive,
            &FilteredReason::ExclusiveTweet,
        );

        let mut super_follow = exclusive_candidate(1, 100, 100);
        super_follow
            .exclusive_content
            .as_mut()
            .unwrap()
            .viewer_super_follows_author = true;
        assert_allows(spec, &viewer(200), &super_follow);

        let reply = exclusive_candidate(2, 200, 100);
        assert_allows(spec, &viewer(200), &reply);

        let mut retweet = exclusive_candidate(2, 200, 100);
        retweet.tweet_features.core.source_tweet_id = Some(99);
        assert_drops(
            spec,
            &viewer(200),
            &retweet,
            &FilteredReason::ExclusiveTweet,
        );
    }

    fn gating_viewer(age: ViewerAge) -> ViewerFeatures {
        ViewerFeatures {
            viewer_age: age,
            country_code: Some("de".into()),
            ..viewer(VIEWER_ID)
        }
    }

    fn media_label(label: SafetyLabelType) -> HydratedTweetCandidate {
        candidate().with_label(label).with_media().build()
    }

    fn no_media_label(label: SafetyLabelType) -> HydratedTweetCandidate {
        let mut c = media_label(label);
        c.tweet_features.media.has_media = false;
        c
    }

    fn nsfw_author_media() -> HydratedTweetCandidate {
        candidate()
            .with_media()
            .with_author_features(AuthorFeatures {
                is_nsfw_user: true,
                ..Default::default()
            })
            .build()
    }

    fn nsfw_tweet_flag_media() -> HydratedTweetCandidate {
        let mut c = nsfw_author_media();
        c.author_features = AuthorFeatures::default();
        c.tweet_features.nsfw = NsfwFeature {
            user: true,
            admin: false,
        };
        c
    }

    fn sensitive_spec(name: &str) -> &'static RuleSpec {
        SENSITIVE_VIEWER_DROPS
            .iter()
            .find(|spec| spec.name() == name)
            .unwrap_or_else(|| panic!("no sensitive-viewer row {name}"))
    }

    fn sensitive_firing_candidates() -> Vec<HydratedTweetCandidate> {
        let mut admin_author = nsfw_author_media();
        admin_author.author_features = AuthorFeatures {
            is_nsfw_admin: true,
            ..Default::default()
        };
        let mut admin_flag = nsfw_tweet_flag_media();
        admin_flag.tweet_features.nsfw = NsfwFeature {
            user: false,
            admin: true,
        };
        let mut both_flags = nsfw_tweet_flag_media();
        both_flags.author_features.is_nsfw_user = true;
        vec![
            media_label(SafetyLabelType::NSFW_HIGH_PRECISION),
            media_label(SafetyLabelType::NSFW_HIGH_RECALL),
            media_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
            no_media_label(SafetyLabelType::NSFW_TEXT),
            no_media_label(SafetyLabelType::NSFW_CARD_IMAGE),
            nsfw_author_media(),
            admin_author,
            nsfw_tweet_flag_media(),
            admin_flag,
            both_flags,
        ]
    }

    #[test]
    fn sensitive_viewer_content_axis() {
        let underage = sensitive_spec("SensitiveViewerUnderageDropRule");
        let logged_out = sensitive_spec("SensitiveViewerLoggedOutDropRule");
        let no_age = sensitive_spec("SensitiveViewerNoStatedAgeDropRule");
        let reason = FilteredReason::ContainNsfwMedia;
        let logged_out_viewer = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..gating_viewer(ViewerAge::Unknown)
        };
        for firing in sensitive_firing_candidates() {
            assert_drops(
                underage,
                &gating_viewer(ViewerAge::Known(15)),
                &firing,
                &reason,
            );
            assert_drops(logged_out, &logged_out_viewer, &firing, &reason);
            assert_drops(
                no_age,
                &gating_viewer(ViewerAge::NotStated),
                &firing,
                &reason,
            );
        }
    }

    #[test]
    fn sensitive_viewer_exemption_axis() {
        let underage = sensitive_spec("SensitiveViewerUnderageDropRule");
        let logged_out = sensitive_spec("SensitiveViewerLoggedOutDropRule");
        let no_age = sensitive_spec("SensitiveViewerNoStatedAgeDropRule");
        let hp = media_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let text = no_media_label(SafetyLabelType::NSFW_TEXT);
        let reason = FilteredReason::ContainNsfwMedia;

        assert_allows(underage, &gating_viewer(ViewerAge::Known(18)), &hp);
        assert_allows(underage, &gating_viewer(ViewerAge::Known(18)), &text);
        assert_allows(underage, &gating_viewer(ViewerAge::Unknown), &hp);
        assert_allows(no_age, &gating_viewer(ViewerAge::Unknown), &hp);
        assert_allows(underage, &gating_viewer(ViewerAge::Unknown), &text);
        assert_allows(no_age, &gating_viewer(ViewerAge::Unknown), &text);

        let opted_in = ViewerFeatures {
            allows_sensitive_media: true,
            ..gating_viewer(ViewerAge::Known(15))
        };
        assert_drops(underage, &opted_in, &hp, &reason);

        let mut self_hp = hp.clone();
        self_hp.author_id = VIEWER_ID;
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &self_hp);
        let mut self_text = text.clone();
        self_text.author_id = VIEWER_ID;
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &self_text);

        let mut hp_no_media = hp.clone();
        hp_no_media.tweet_features.media.has_media = false;
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &hp_no_media);
        let logged_out_viewer = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..gating_viewer(ViewerAge::Unknown)
        };
        assert_allows(logged_out, &logged_out_viewer, &hp_no_media);
        assert_allows(logged_out, &gating_viewer(ViewerAge::Known(15)), &hp);

        let mut no_flags = nsfw_tweet_flag_media();
        no_flags.tweet_features.nsfw = NsfwFeature::default();
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &no_flags);

        let mut flag_rt = nsfw_tweet_flag_media();
        flag_rt.tweet_features.core.source_tweet_id = Some(42);
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &flag_rt);
        let mut flag_self = nsfw_tweet_flag_media();
        flag_self.author_id = VIEWER_ID;
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &flag_self);

        let mut author_rt = nsfw_author_media();
        author_rt.tweet_features.core.source_tweet_id = Some(42);
        assert_allows(underage, &gating_viewer(ViewerAge::Known(15)), &author_rt);
        let mut author_no_media = nsfw_author_media();
        author_no_media.tweet_features.media.has_media = false;
        assert_allows(
            underage,
            &gating_viewer(ViewerAge::Known(15)),
            &author_no_media,
        );
    }

    fn tes_spec(name: &str) -> &'static RuleSpec {
        TES_HOME_DROPS
            .iter()
            .chain(RECS_MEDIA_DROPS)
            .find(|spec| spec.name() == name)
            .unwrap_or_else(|| panic!("no TES row {name}"))
    }

    fn stale_edit_control() -> Option<EditControl> {
        Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![1, 2],
            ..Default::default()
        }))
    }

    fn takedown_candidate(reasons: Vec<TakedownReason>) -> HydratedTweetCandidate {
        candidate()
            .with_tweet_features(TweetFeatures {
                takedown_reasons: reasons,
                ..Default::default()
            })
            .build()
    }

    fn viewer_with_country(country: &str) -> ViewerFeatures {
        ViewerFeatures {
            country_code: Some(country.to_string()),
            ..viewer(VIEWER_ID)
        }
    }

    fn geo_candidate(allow: &[&str], deny: &[&str]) -> HydratedTweetCandidate {
        candidate()
            .with_tweet_features(TweetFeatures {
                media: MediaFeature {
                    geo_allow_list: allow.iter().map(|s| s.to_string()).collect(),
                    geo_deny_list: deny.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                },
                ..Default::default()
            })
            .build()
    }

    #[test]
    fn filter_all_axis() {
        let spec = &FILTER_ALL[0];
        let RuleSpec::Tweet {
            name: "FilterAllRule",
            action: RuleAction::Drop(reason),
            exempt_author: false,
            ..
        } = spec
        else {
            panic!("{} is not the FilterAll row", spec.name());
        };
        let pristine = candidate().build();
        assert_drops(spec, &viewer(VIEWER_ID), &pristine, reason);
        assert_drops(spec, &author_viewer(), &pristine, reason);
        assert_drops(spec, &logged_out_viewer(), &pristine, reason);
    }

    #[test]
    fn stale_and_dmca_tweet_axis() {
        let stale = tes_spec("DropStaleTweetsRule");
        let dmca = tes_spec("DropTweetsWithDmcaMediaRule");
        let reason = FilteredReason::UnspecifiedReason;
        let stale_c = candidate()
            .with_tweet_features(TweetFeatures {
                edit_control: stale_edit_control(),
                ..Default::default()
            })
            .build();
        assert_drops(stale, &viewer(VIEWER_ID), &stale_c, &reason);
        assert_allows(stale, &viewer(VIEWER_ID), &candidate().build());
        let stale_rt = candidate()
            .with_tweet_features(TweetFeatures {
                edit_control: stale_edit_control(),
                ..Default::default()
            })
            .retweet_of(99)
            .build();
        assert_allows(stale, &viewer(VIEWER_ID), &stale_rt);

        let dmca_c = candidate()
            .with_tweet_features(TweetFeatures {
                media: MediaFeature {
                    has_dmca_media: true,
                    ..Default::default()
                },
                ..Default::default()
            })
            .build();
        assert_drops(dmca, &viewer(VIEWER_ID), &dmca_c, &reason);
        assert_allows(dmca, &viewer(VIEWER_ID), &candidate().build());
    }

    #[test]
    fn takedown_country_axis() {
        let legal = tes_spec("DropLegalTakendownPostRule");
        let local = tes_spec("DropLocalLawsTakendownPostRule");
        let reason = FilteredReason::UnspecifiedReason;
        let legal_c = takedown_candidate(vec![
            TakedownReason::LegalRequest {
                country_code: "de".to_string(),
            },
            TakedownReason::UnspecifiedReason {
                country_code: "fr".to_string(),
            },
        ]);
        assert_drops(legal, &viewer_with_country("de"), &legal_c, &reason);
        assert_allows(legal, &viewer_with_country("us"), &legal_c);
        assert_allows(legal, &viewer(VIEWER_ID), &legal_c);

        let bystander = takedown_candidate(vec![TakedownReason::BystanderReport {
            country_code: "de".to_string(),
        }]);
        assert_allows(legal, &viewer_with_country("de"), &bystander);
        assert_drops(local, &viewer_with_country("de"), &bystander, &reason);
        assert_allows(local, &viewer_with_country("us"), &bystander);

        let legal_only = takedown_candidate(vec![TakedownReason::LegalRequest {
            country_code: "de".to_string(),
        }]);
        assert_allows(local, &viewer_with_country("de"), &legal_only);

        let mut author_legal = legal_only.clone();
        author_legal.author_id = VIEWER_ID;
        assert_allows(legal, &viewer_with_country("de"), &author_legal);
        let mut author_local = bystander.clone();
        author_local.author_id = VIEWER_ID;
        assert_allows(local, &viewer_with_country("de"), &author_local);

        let non_country = takedown_candidate(vec![
            TakedownReason::HatefulImagery,
            TakedownReason::Unknown,
        ]);
        assert_allows(legal, &viewer_with_country("de"), &non_country);
        assert_allows(local, &viewer_with_country("de"), &non_country);
    }

    #[test]
    fn takedown_worldwide_axis() {
        let legal = tes_spec("DropLegalTakendownPostRule");
        let local = tes_spec("DropLocalLawsTakendownPostRule");
        let reason = FilteredReason::UnspecifiedReason;

        for code in ["xx", "xy", "XX"] {
            let worldwide = takedown_candidate(vec![TakedownReason::LegalRequest {
                country_code: code.to_string(),
            }]);
            assert_drops(legal, &viewer_with_country("us"), &worldwide, &reason);
            assert_drops(legal, &viewer(VIEWER_ID), &worldwide, &reason);
        }

        let bystander_worldwide = takedown_candidate(vec![TakedownReason::BystanderReport {
            country_code: "xx".to_string(),
        }]);
        assert_drops(
            local,
            &viewer_with_country("us"),
            &bystander_worldwide,
            &reason,
        );
        assert_drops(local, &viewer(VIEWER_ID), &bystander_worldwide, &reason);

        let country_scoped = takedown_candidate(vec![TakedownReason::LegalRequest {
            country_code: "de".to_string(),
        }]);
        assert_allows(legal, &viewer(VIEWER_ID), &country_scoped);
        let bystander_scoped = takedown_candidate(vec![TakedownReason::BystanderReport {
            country_code: "de".to_string(),
        }]);
        assert_allows(local, &viewer(VIEWER_ID), &bystander_scoped);

        let dmca = takedown_candidate(vec![TakedownReason::Dmca]);
        assert_drops(legal, &viewer_with_country("de"), &dmca, &reason);
        assert_drops(legal, &viewer(VIEWER_ID), &dmca, &reason);
        assert_allows(local, &viewer_with_country("de"), &dmca);
        let mut author_dmca = dmca.clone();
        author_dmca.author_id = VIEWER_ID;
        assert_allows(legal, &viewer(VIEWER_ID), &author_dmca);
    }

    #[test]
    fn geo_restricted_media_axis() {
        let spec = tes_spec("DropTweetsWithGeoRestrictedMediaRule");
        let reason = FilteredReason::UnspecifiedReason;
        assert_allows(spec, &viewer_with_country("us"), &geo_candidate(&[], &[]));
        assert_drops(
            spec,
            &viewer_with_country("de"),
            &geo_candidate(&[], &["de", "fr"]),
            &reason,
        );
        assert_allows(
            spec,
            &viewer_with_country("us"),
            &geo_candidate(&[], &["de", "fr"]),
        );
        assert_allows(
            spec,
            &viewer_with_country("us"),
            &geo_candidate(&["us", "gb"], &[]),
        );
        assert_drops(
            spec,
            &viewer_with_country("de"),
            &geo_candidate(&["us", "gb"], &[]),
            &reason,
        );
        assert_allows(
            spec,
            &viewer_with_country("us"),
            &geo_candidate(&["US"], &[]),
        );
        assert_drops(
            spec,
            &viewer_with_country("de"),
            &geo_candidate(&[], &["DE"]),
            &reason,
        );
        assert_drops(
            spec,
            &viewer(VIEWER_ID),
            &geo_candidate(&["us"], &[]),
            &reason,
        );
        assert_drops(
            spec,
            &viewer(VIEWER_ID),
            &geo_candidate(&[], &["xx"]),
            &reason,
        );
        assert_allows(spec, &viewer(VIEWER_ID), &geo_candidate(&[], &["de"]));

        let mut author = geo_candidate(&[], &["de"]);
        author.author_id = VIEWER_ID;
        assert_drops(spec, &viewer_with_country("de"), &author, &reason);

        let mut retweet = geo_candidate(&[], &["de"]);
        retweet.tweet_features.core.source_tweet_id = Some(99);
        assert_drops(spec, &viewer_with_country("de"), &retweet, &reason);
    }

    #[test]
    fn nullcast_drop_axis() {
        let spec = &NULLCAST_DROP[0];
        let RuleSpec::Tweet {
            name: "NullcastedTweetDropRule",
            action: RuleAction::Drop(reason),
            exempt_author: false,
            ..
        } = spec
        else {
            panic!("{} is not the nullcast drop row", spec.name());
        };
        let firing = candidate()
            .with_tweet_features(TweetFeatures {
                is_nullcast: true,
                ..Default::default()
            })
            .build();
        assert_drops(spec, &viewer(VIEWER_ID), &firing, reason);
        assert_drops(spec, &author_viewer(), &firing, reason);
        assert_allows(spec, &viewer(VIEWER_ID), &candidate().build());

        let mut community = firing.clone();
        community.tweet_features.is_community_tweet = true;
        assert_allows(spec, &viewer(VIEWER_ID), &community);

        let mut retweet = firing.clone();
        retweet.tweet_features.core.source_tweet_id = Some(99);
        assert_allows(spec, &viewer(VIEWER_ID), &retweet);
    }

    #[test]
    fn no_stated_age_jurisdiction_axis() {
        let no_age = sensitive_spec("SensitiveViewerNoStatedAgeDropRule");
        let hp = media_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        let text = no_media_label(SafetyLabelType::NSFW_TEXT);
        let reason = FilteredReason::ContainNsfwMedia;

        let us = ViewerFeatures {
            country_code: Some("us".into()),
            ..gating_viewer(ViewerAge::NotStated)
        };
        assert_allows(no_age, &us, &hp);
        assert_allows(no_age, &us, &text);

        let missing = ViewerFeatures {
            country_code: None,
            ..gating_viewer(ViewerAge::NotStated)
        };
        assert_allows(no_age, &missing, &hp);

        let account_overrides = ViewerFeatures {
            country_code: Some("de".into()),
            account_country_code: Some("us".into()),
            ..gating_viewer(ViewerAge::NotStated)
        };
        assert_allows(no_age, &account_overrides, &hp);

        let gating_account = ViewerFeatures {
            country_code: Some("us".into()),
            account_country_code: Some("kr".into()),
            ..gating_viewer(ViewerAge::NotStated)
        };
        assert_drops(no_age, &gating_account, &hp, &reason);

        let request_fallback = ViewerFeatures {
            country_code: Some("de".into()),
            account_country_code: None,
            ..gating_viewer(ViewerAge::NotStated)
        };
        assert_drops(no_age, &request_fallback, &hp, &reason);
    }

    fn all_rule_slices() -> [&'static [RuleSpec]; 16] {
        use crate::rules::author_rules::{
            AUTHOR_STATE_DROPS, OON_NSFW_AUTHOR_DROPS, OON_USER_LABEL_DROPS, SOCIALGRAPH_DROPS,
        };
        [
            SAFETY_HYDRATION_FAILURE_DROP,
            AUTHOR_STATE_DROPS,
            TWEET_LABEL_DROPS,
            NSFW_MEDIA_INTERSTITIALS,
            OON_NSFW_AUTHOR_DROPS,
            OON_TWEET_FLAG_DROPS,
            OON_TWEET_LABEL_DROPS,
            OON_USER_LABEL_DROPS,
            SOCIALGRAPH_DROPS,
            EXCLUSIVE_TWEET_DROP,
            NSFW_AUTHOR_INTERSTITIAL,
            NULLCAST_DROP,
            TES_HOME_DROPS,
            FILTER_ALL,
            RECS_MEDIA_DROPS,
            SENSITIVE_VIEWER_DROPS,
        ]
    }

    #[test]
    fn wired_rule_names_are_unique_and_nonempty() {
        let mut seen = std::collections::BTreeSet::new();
        for spec in all_rule_slices().into_iter().flatten() {
            let name = spec.name();
            assert!(!name.is_empty(), "rule name must be non-empty");
            assert!(seen.insert(name), "duplicate wired rule name {name}");
        }
    }
}
