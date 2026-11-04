use super::{Role, Row, Subject};
use crate::models::{
    EmergencyCopy, LabelInterstitial, LimitedEngagementReason, PublicInterest, SafetyLabelType,
};
use crate::rules::fixtures::{
    allow, blurred, candidate, dropped, label_interstitial, legacy_interstitial, on_client,
    read_only_viewer, viewer, with_label_interstitial, VIEWER_ID,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::InterstitialReason;
use xai_x_thrift::safety_result::SafetyResultReason;

const POSTED_2020: u64 = (1577836800000 - 1288834974657) << 22;
const POSTED_2018: u64 = (1514764800000 - 1288834974657) << 22;

fn public_interest(
    reason: SafetyResultReason,
    limit: Option<LimitedEngagementReason>,
) -> LabelInterstitial {
    LabelInterstitial::PublicInterest(PublicInterest { reason, limit })
}

fn one_off_notice() -> LabelInterstitial {
    public_interest(
        SafetyResultReason::ONE_OFF,
        Some(LimitedEngagementReason::NonCompliant),
    )
}

pub(super) fn rows() -> Vec<Row> {
    let one_off =
        || label_interstitial(one_off_notice(), "public_interest/label_interstitial", None);
    vec![
        Row {
            name: "low_quality_label_one_off_source",
            subject: Subject::Post(
                candidate()
                    .tweet_id(POSTED_2020)
                    .with_label_source(SafetyLabelType::LOW_QUALITY, "OneOff")
                    .build(),
            ),
            expect: vec![
                (Role::each([Role::NonFollower, Role::Author]), one_off()),
                (
                    Role::As("read_only", read_only_viewer(VIEWER_ID)),
                    label_interstitial(
                        public_interest(
                            SafetyResultReason::ONE_OFF,
                            Some(LimitedEngagementReason::NonCompliant),
                        ),
                        "public_interest/label_interstitial",
                        Some((
                            LimitedEngagementReason::ReadonlyViewer,
                            "read_only_viewer/limited_engagement",
                        )),
                    ),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "low_quality_label_episodic_source",
            subject: Subject::Post(
                candidate()
                    .tweet_id(POSTED_2020)
                    .with_label_source(SafetyLabelType::LOW_QUALITY, "AbuseEpisodicHatefulConduct")
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                label_interstitial(
                    public_interest(
                        SafetyResultReason::ABUSE_EPISODIC_HATEFUL_CONDUCT,
                        Some(LimitedEngagementReason::NonCompliant),
                    ),
                    "public_interest/label_interstitial",
                    None,
                ),
            )],
            at: vec![],
        },
        Row {
            name: "low_quality_label_with_sensitive_and_violent_media",
            subject: Subject::Post(
                candidate()
                    .tweet_id(POSTED_2020)
                    .with_label_source(SafetyLabelType::LOW_QUALITY, "OneOff")
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION)
                    .build(),
            ),
            expect: vec![(
                Role::As("no_client_context", viewer(VIEWER_ID)),
                with_label_interstitial(
                    legacy_interstitial("nsfw_high_precision/legacy_interstitial"),
                    one_off_notice(),
                    "public_interest/label_interstitial",
                ),
            )],
            at: vec![(
                TimelineHomeHydration,
                Role::As("web_in_us", on_client("web", "us", viewer(VIEWER_ID))),
                with_label_interstitial(
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_high_precision/blur/sensitive",
                    ),
                    one_off_notice(),
                    "public_interest/label_interstitial",
                ),
            )],
        },
        Row {
            name: "misinfo_civic_label_limiting_replies",
            subject: Subject::Post(
                candidate()
                    .tweet_id(POSTED_2020)
                    .with_label_source(SafetyLabelType::MISINFO_CIVIC, "limited_replies")
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                label_interstitial(
                    public_interest(
                        SafetyResultReason::MISINFO_CIVIC,
                        Some(LimitedEngagementReason::ConversationControl),
                    ),
                    "public_interest/label_interstitial",
                    None,
                ),
            )],
            at: vec![],
        },
        Row {
            name: "low_quality_label_before_public_interest_labels",
            subject: Subject::Post(
                candidate()
                    .tweet_id(POSTED_2018)
                    .with_label_source(SafetyLabelType::LOW_QUALITY, "OneOff")
                    .build(),
            ),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "misinfo_civic_label_without_source",
            subject: Subject::Post(
                candidate()
                    .tweet_id(POSTED_2020)
                    .with_label(SafetyLabelType::MISINFO_CIVIC)
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                label_interstitial(
                    public_interest(SafetyResultReason::MISINFO_CIVIC, None),
                    "public_interest/label_interstitial",
                    None,
                ),
            )],
            at: vec![],
        },
        Row {
            name: "for_emergency_use_only_label_with_copy",
            subject: Subject::Post(
                candidate()
                    .with_label_source(
                        SafetyLabelType::FOR_EMERGENCY_USE_ONLY,
                        "Stay safe.|https://help.x.com/emergency",
                    )
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                label_interstitial(
                    LabelInterstitial::EmergencyDynamic(Box::new(EmergencyCopy {
                        copy: "Stay safe.".to_owned(),
                        link: Some("https://help.x.com/emergency".to_owned()),
                    })),
                    "for_emergency_use_only/label_interstitial",
                    None,
                ),
            )],
            at: vec![(
                TimelineHome,
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "for_emergency_use_only/drop/unspecified",
                ),
            )],
        },
        Row {
            name: "for_emergency_use_only_label_with_rule_source",
            subject: Subject::Post(
                candidate()
                    .with_label_source(SafetyLabelType::FOR_EMERGENCY_USE_ONLY, "bot_id_7")
                    .build(),
            ),
            expect: vec![],
            at: vec![(TimelineHomeHydration, Role::NonFollower, allow())],
        },
    ]
}
