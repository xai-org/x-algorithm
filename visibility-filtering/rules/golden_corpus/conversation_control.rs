use super::{Role, Row, Subject};
use crate::hydration::Hydrator::{
    RootFollowsViewer, RootFollowsViewerSecondDegree, SuperFollowsRoot, ViewerFollowsRoot,
};
use crate::models::{
    ConversationControlFeatures, LimitedEngagementReason, PostCandidate, SafetyLabelType,
    ViewerProfile,
};
use crate::rules::fixtures::{
    allow, blurred, blurred_and_limited, candidate, controlled_root, conversation_control, limited,
    limited_for, on_client, read_only_viewer, viewer, viewer_with_profile, VIEWER_ID,
};
use crate::rules::SafetyLevel::TimelineHomeHydration;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};
use xai_x_thrift::action::InterstitialReason;

const REPLY_ROOT_AUTHOR_ID: u64 = 4242;

pub(super) fn rows() -> Vec<Row> {
    use ConversationControlArm::{
        ByInvitation, Co, Community, Followers, MyNetwork, Subscribers, Verified,
    };
    let limits = |rule| limited(LimitedEngagementReason::ConversationControl, rule);
    [
        (
            ByInvitation,
            vec![
                (
                    Role::NonFollower,
                    limits("limit_replies_by_invitation/limited_engagement/conversation_control"),
                ),
                (
                    Role::As("read_only", read_only_viewer(VIEWER_ID)),
                    limited_for(
                        &[
                            LimitedEngagementReason::ConversationControl,
                            LimitedEngagementReason::ReadonlyViewer,
                        ],
                        "limit_replies_by_invitation/limited_engagement/conversation_control",
                    ),
                ),
            ],
            [
                "by_invitation_conversation",
                "by_invitation_conversation_invited",
                "by_invitation_conversation_retweet",
            ],
        ),
        (
            Community,
            vec![(
                Role::NonFollower,
                limits("limit_replies_community/limited_engagement/conversation_control"),
            )],
            [
                "community_conversation",
                "community_conversation_invited",
                "community_conversation_retweet",
            ],
        ),
        (
            Followers,
            vec![(
                Role::NonFollower,
                limits("limit_replies_followers/limited_engagement/conversation_control"),
            )],
            [
                "followers_conversation",
                "followers_conversation_invited",
                "followers_conversation_retweet",
            ],
        ),
        (
            Subscribers,
            vec![(
                Role::NonFollower,
                limits("limit_replies_subscribers/limited_engagement/conversation_control"),
            )],
            [
                "subscribers_conversation",
                "subscribers_conversation_invited",
                "subscribers_conversation_retweet",
            ],
        ),
        (
            Verified,
            vec![
                (
                    Role::NonFollower,
                    limits("limit_replies_verified/limited_engagement/conversation_control"),
                ),
                (
                    Role::As(
                        "verified",
                        viewer_with_profile(ViewerProfile {
                            has_verified_badge: true,
                            ..ViewerProfile::default()
                        }),
                    ),
                    allow(),
                ),
            ],
            [
                "verified_conversation",
                "verified_conversation_invited",
                "verified_conversation_retweet",
            ],
        ),
        (
            MyNetwork,
            vec![(
                Role::NonFollower,
                limits("limit_replies_my_network/limited_engagement/conversation_control"),
            )],
            [
                "my_network_conversation",
                "my_network_conversation_invited",
                "my_network_conversation_retweet",
            ],
        ),
        (
            Co,
            vec![],
            [
                "co_conversation",
                "co_conversation_invited",
                "co_conversation_retweet",
            ],
        ),
    ]
    .into_iter()
    .flat_map(|(arm, mut root_expect, [root, invited, retweet])| {
        root_expect.push((Role::Author, allow()));
        root_expect.push((Role::LoggedOut, allow()));
        [
            Row {
                name: root,
                subject: Subject::Post(controlled_candidate(controlled_root(arm))),
                expect: root_expect,
                at: vec![],
            },
            Row {
                name: invited,
                subject: Subject::Post(controlled_candidate(ConversationControlFeatures {
                    control: ConversationControl {
                        invited_user_ids: vec![VIEWER_ID],
                        ..controlled_root(arm).control
                    },
                    ..controlled_root(arm)
                })),
                expect: vec![(Role::NonFollower, allow())],
                at: vec![],
            },
            Row {
                name: retweet,
                subject: Subject::Post(
                    candidate()
                        .retweet_of(2)
                        .with_conversation_control(controlled_root(arm))
                        .build(),
                ),
                expect: vec![(Role::NonFollower, allow())],
                at: vec![],
            },
        ]
    })
    .chain([
        Row {
            name: "nsfw_by_invitation_conversation",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_media()
                    .with_conversation_control(controlled_root(ByInvitation))
                    .build(),
            ),
            expect: vec![],
            at: vec![(
                TimelineHomeHydration,
                Role::As("web_in_us", on_client("web", "us", viewer(VIEWER_ID))),
                blurred_and_limited(
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_high_precision/blur/sensitive",
                    ),
                    limits("limit_replies_by_invitation/limited_engagement/conversation_control"),
                ),
            )],
        },
        Row {
            name: "dynamic_product_ad_by_invitation_conversation",
            subject: Subject::Post(
                candidate()
                    .with_label(SafetyLabelType::DYNAMIC_PRODUCT_AD)
                    .with_conversation_control(controlled_root(ByInvitation))
                    .build(),
            ),
            expect: vec![(
                Role::NonFollower,
                limited_for(
                    &[
                        LimitedEngagementReason::ConversationControl,
                        LimitedEngagementReason::DynamicProductAd,
                    ],
                    "limit_replies_by_invitation/limited_engagement/conversation_control",
                ),
            )],
            at: vec![],
        },
        Row {
            name: "by_invitation_reply",
            subject: Subject::Post(controlled_candidate(conversation_control(
                ByInvitation,
                REPLY_ROOT_AUTHOR_ID,
            ))),
            expect: vec![(
                Role::Author,
                limits("limit_replies_by_invitation/limited_engagement/conversation_control"),
            )],
            at: vec![],
        },
        Row {
            name: "community_conversation_followed_viewer",
            subject: Subject::Post(
                candidate()
                    .with_edge(RootFollowsViewer)
                    .with_conversation_control(controlled_root(Community))
                    .build(),
            ),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "followers_conversation_follower",
            subject: Subject::Post(
                candidate()
                    .with_edge(ViewerFollowsRoot)
                    .with_conversation_control(controlled_root(Followers))
                    .build(),
            ),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "my_network_conversation_followed_viewer",
            subject: Subject::Post(
                candidate()
                    .with_edge(RootFollowsViewer)
                    .with_conversation_control(controlled_root(MyNetwork))
                    .build(),
            ),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "my_network_conversation_second_degree_viewer",
            subject: Subject::Post(
                candidate()
                    .with_edge(RootFollowsViewerSecondDegree)
                    .with_conversation_control(controlled_root(MyNetwork))
                    .build(),
            ),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "subscribers_conversation_super_follower",
            subject: Subject::Post(
                candidate()
                    .with_edge(SuperFollowsRoot)
                    .with_conversation_control(controlled_root(Subscribers))
                    .build(),
            ),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "co_conversation_outside_allowed_countries",
            subject: Subject::Post(controlled_candidate(co_root(&["EUR", "BR"], Some("US")))),
            expect: vec![(
                Role::NonFollower,
                limits("limit_replies_co/limited_engagement/conversation_control"),
            )],
            at: vec![],
        },
        Row {
            name: "co_conversation_unknown_viewer_country",
            subject: Subject::Post(controlled_candidate(co_root(&["US"], None))),
            expect: vec![(
                Role::NonFollower,
                limits("limit_replies_co/limited_engagement/conversation_control"),
            )],
            at: vec![],
        },
        Row {
            name: "co_conversation_no_allowed_country",
            subject: Subject::Post(controlled_candidate(co_root(&[], Some("US")))),
            expect: vec![(
                Role::NonFollower,
                limits("limit_replies_co/limited_engagement/conversation_control"),
            )],
            at: vec![],
        },
        Row {
            name: "co_conversation_allowed_country",
            subject: Subject::Post(controlled_candidate(co_root(&["BR"], Some("br")))),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
        Row {
            name: "co_conversation_allowed_region",
            subject: Subject::Post(controlled_candidate(co_root(&["NAM"], Some("US")))),
            expect: vec![(Role::NonFollower, allow())],
            at: vec![],
        },
    ])
    .collect()
}

fn co_root(
    allowed_country_codes: &[&str],
    viewer_country: Option<&str>,
) -> ConversationControlFeatures {
    ConversationControlFeatures {
        control: ConversationControl {
            allowed_country_codes: allowed_country_codes
                .iter()
                .map(|code| (*code).to_string())
                .collect(),
            ..controlled_root(ConversationControlArm::Co).control
        },
        viewer_country: viewer_country.map(Into::into),
    }
}

fn controlled_candidate(features: ConversationControlFeatures) -> PostCandidate {
    candidate().with_conversation_control(features).build()
}
