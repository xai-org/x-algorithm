use super::{Role, Row, Subject};
use crate::hydration::Hydrator;
use crate::models::{
    ClientCapability, ClientSwitch, RequestFacts, SafetyLabelType, ViewerFeatures, ViewerProfile,
};
use crate::rules::fixtures::{
    allow, appealed, blurred, candidate, controlled_root, dropped, fosnr, labeled, on_client,
    viewer, viewer_in_country, viewer_with_profile, AUTHOR_ID, VIEWER_ID,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration, TimelineHomeRecommendations};
use xai_core_entities::entities::ConversationControlArm;
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{AppealablePolicy, InterstitialReason};

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "malicious_url_label",
            subject: Subject::Post(labeled(SafetyLabelType::MALICIOUS_URL)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "malicious_url/drop/undesirable",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "nsfw_high_recall_label",
            subject: Subject::Post(labeled(SafetyLabelType::NSFW_HIGH_RECALL)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_high_recall/drop/nsfw_media",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "do_not_amplify_label",
            subject: Subject::Post(labeled(SafetyLabelType::DO_NOT_AMPLIFY)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "do_not_amplify/drop/undesirable",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![],
        },
        Row {
            name: "spam_high_recall_label",
            subject: Subject::Post(labeled(SafetyLabelType::SPAM_HIGH_RECALL)),
            expect: vec![
                (
                    Role::each([Role::NonFollower, Role::Follower]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "spam_high_recall/drop/undesirable",
                    ),
                ),
                (Role::Author, allow()),
            ],
            at: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "brazil_election_legal_label",
            subject: Subject::Post(labeled(SafetyLabelType::BRAZIL_ELECTION_LEGAL)),
            expect: vec![
                (
                    Role::As("in_br", viewer_in_country("br")),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "brazil_election_legal/drop/undesirable",
                    ),
                ),
                (
                    Role::each([
                        Role::As("in_us", viewer_in_country("us")),
                        Role::As(
                            "account_br_no_request_country",
                            viewer_with_profile(ViewerProfile {
                                account_country_code: Some("br".to_string()),
                                ..ViewerProfile::default()
                            }),
                        ),
                        Role::As(
                            "author_in_br",
                            ViewerFeatures {
                                facts: RequestFacts::default().with_country("br"),
                                ..viewer(AUTHOR_ID)
                            },
                        ),
                        Role::NonFollower,
                    ]),
                    allow(),
                ),
            ],
            at: vec![(
                TimelineHome,
                Role::As("in_br", viewer_in_country("br")),
                allow(),
            )],
        },
        Row {
            name: "brazil_election_legal_label_followed",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::BRAZIL_ELECTION_LEGAL)
                    .with_edge(Hydrator::Follows)
                    .build(),
            ),
            expect: vec![(
                Role::As("follower_in_br", viewer_in_country("br")),
                dropped(
                    FilteredReason::PossiblyUndesirable,
                    "brazil_election_legal/drop/undesirable",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "fosnr_abuse_insults_label",
            subject: Subject::Post(labeled(SafetyLabelType::FOSNR_ABUSE_INSULTS)),
            expect: vec![
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::ABUSE,
                        1,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
                (
                    Role::each([Role::NonFollower, Role::LoggedOut]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults_non_follower/drop/undesirable",
                    ),
                ),
                (
                    Role::Follower,
                    fosnr(
                        true,
                        false,
                        "fosnr_abuse_insults_follower/soft_intervention/abuse",
                    ),
                ),
                (
                    Role::each([
                        Role::As("client_without_fosnr", client_without_fosnr(VIEWER_ID)),
                        Role::As(
                            "author_on_client_without_fosnr",
                            client_without_fosnr(AUTHOR_ID),
                        ),
                    ]),
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_fallback/drop/undesirable",
                    ),
                ),
            ],
            at: vec![
                (TimelineHome, Role::NonFollower, allow()),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults/drop/undesirable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_abuse_insults_by_agent_under_appeal",
            subject: Subject::Post(
                candidate()
                    .with_agent_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                    .with_label(SafetyLabelType::FOSNR_APPEAL_SUBMITTED)
                    .build(),
            ),
            expect: vec![
                (
                    Role::Follower,
                    fosnr(
                        false,
                        true,
                        "fosnr_abuse_insults_follower/soft_intervention/abuse",
                    ),
                ),
                (
                    Role::Author,
                    appealed(
                        AppealablePolicy::ABUSE,
                        1,
                        false,
                        true,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
            at: vec![],
        },
        Row {
            name: "fosnr_abuse_insults_on_nsfw_media",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_media()
                    .with_edge(Hydrator::Follows)
                    .build(),
            ),
            expect: vec![],
            at: vec![(
                TimelineHomeHydration,
                Role::As(
                    "web_follower_in_us",
                    on_client("web", "us", viewer(VIEWER_ID)),
                ),
                blurred(
                    InterstitialReason::Sensitive(true),
                    "nsfw_high_precision/blur/sensitive",
                ),
            )],
        },
        Row {
            name: "fosnr_abuse_insults_in_by_invitation_conversation",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                    .with_conversation_control(controlled_root(
                        ConversationControlArm::ByInvitation,
                    ))
                    .build(),
            ),
            expect: vec![(
                Role::Follower,
                fosnr(
                    true,
                    false,
                    "fosnr_abuse_insults_follower/soft_intervention/abuse",
                ),
            )],
            at: vec![],
        },
    ]
}

fn client_without_fosnr(viewer_id: u64) -> ViewerFeatures {
    ViewerFeatures {
        facts: RequestFacts::default().with_client(
            ClientCapability::DEFAULT
                .without(ClientSwitch::FosnrRules)
                .with(ClientSwitch::FosnrFallbackDrops),
        ),
        ..viewer(viewer_id)
    }
}
