use super::{Role, Row, Subject};
use crate::hydration::Hydrator;
use crate::models::{LimitedEngagementReason, NsfwFeature, SafetyLabelType, TweetFeatures};
use crate::rules::fixtures::{
    allow, blurred, blurred_and_limited, candidate, labeled, legacy_interstitial, limited,
    sensitive_opt_in_viewer, viewer, VIEWER_ID,
};
use crate::rules::SafetyLevel::TimelineHome;
use xai_x_thrift::action::InterstitialReason;

fn no_client_context() -> Role {
    Role::As("no_client_context", viewer(VIEWER_ID))
}

pub(super) fn rows() -> Vec<Row> {
    let high_precision_fallback = || legacy_interstitial("nsfw_high_precision/legacy_interstitial");
    let account_fallback = || legacy_interstitial("nsfw_account/legacy_interstitial");
    vec![
        Row {
            name: "nsfw_high_precision_label_without_the_modern_blur",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_HIGH_PRECISION)),
            expect: vec![(
                Role::each([
                    no_client_context(),
                    Role::Author,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                ]),
                high_precision_fallback(),
            )],
            at: vec![(
                TimelineHome,
                no_client_context(),
                blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_high_precision/blur/sensitive",
                ),
            )],
        },
        Row {
            name: "nsfw_high_precision_label_from_a_blocking_author_without_the_modern_blur",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_edge(Hydrator::BlockedByAuthor)
                    .build(),
            ),
            expect: vec![(
                no_client_context(),
                blurred_and_limited(
                    high_precision_fallback(),
                    limited(
                        LimitedEngagementReason::BlockedViewer,
                        "blocked_viewer/limited_engagement",
                    ),
                ),
            )],
            at: vec![],
        },
        Row {
            name: "gore_and_violence_label_without_the_modern_blur",
            subject: Subject::Post(labeled(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)),
            expect: vec![(
                no_client_context(),
                legacy_interstitial("gore_and_violence_high_precision/legacy_interstitial"),
            )],
            at: vec![],
        },
        Row {
            name: "nsfw_card_image_label_without_the_modern_blur",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_CARD_IMAGE)),
            expect: vec![(
                no_client_context(),
                legacy_interstitial("nsfw_card_image/legacy_interstitial"),
            )],
            at: vec![],
        },
        Row {
            name: "nsfw_admin_flag_media_without_the_modern_blur",
            subject: Subject::Post(
                candidate()
                    .with_tweet_features(TweetFeatures {
                        nsfw: NsfwFeature {
                            admin: true,
                            user: false,
                        },
                        ..TweetFeatures::default()
                    })
                    .with_media()
                    .build(),
            ),
            expect: vec![
                (
                    Role::each([no_client_context(), Role::Author]),
                    account_fallback(),
                ),
                (
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
            ],
            at: vec![],
        },
    ]
}
