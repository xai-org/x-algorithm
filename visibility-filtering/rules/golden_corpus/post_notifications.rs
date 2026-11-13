use super::notifications::{post_kind_cases, recipient};
use super::{CorpusCase, Judging, Subject};
use crate::hydration::Hydrator;
use crate::models::{
    ArticleLifecycle, AuthorFeatures, CommunityModeration, NotificationKind, NsfwViewerDropReason,
    Placement, PostCandidate, SafetyLabelType, TweetFeatures, Verdict, ViewerAge, ViewerFeatures,
    ViewerProfile,
};
use crate::rules::fixtures::{
    allow, candidate, dropped, labeled, logged_out_viewer, no_stated_age_viewer,
    nsfw_high_precision_media, nsfw_viewer_dropped, viewer_in_country, viewer_with_age,
    CandidateBuilder, AUTHOR_ID, VIEWER_ID,
};
use crate::rules::SafetyLevel;
use std::num::NonZeroU64;
use strum::VariantArray;
use xai_core_entities::entities::TakedownReason;
use xai_visibility_filtering::models::{
    Action, DropReason, FilteredReason, SafetyResult, SafetyResultReason,
};

const AFTER_LOW_QUALITY_PROXY: u64 = ((1554076800000 - 1288834974657) << 22) + (1 << 22);

struct Case {
    name: &'static str,
    role: &'static str,
    recipient: ViewerFeatures,
    post: PostCandidate,
    mention_and_reply: Verdict,
    media_tagged: Verdict,
    retweet_source: Verdict,
}

fn shown(
    name: &'static str,
    role: &'static str,
    recipient: ViewerFeatures,
    post: PostCandidate,
) -> Case {
    Case {
        name,
        role,
        recipient,
        post,
        mention_and_reply: allow(),
        media_tagged: allow(),
        retweet_source: allow(),
    }
}

fn dropped_at_79(
    name: &'static str,
    role: &'static str,
    recipient: ViewerFeatures,
    post: PostCandidate,
    verdict: Verdict,
) -> Case {
    Case {
        mention_and_reply: verdict.clone(),
        retweet_source: verdict,
        ..shown(name, role, recipient, post)
    }
}

fn default_recipient() -> ViewerFeatures {
    recipient(VIEWER_ID, ViewerProfile::default())
}

fn unspecified(by: &'static str) -> Verdict {
    dropped(FilteredReason::UnspecifiedReason, by)
}

fn undesirable(by: &'static str) -> Verdict {
    dropped(FilteredReason::PossiblyUndesirable, by)
}

fn tweet(features: TweetFeatures) -> CandidateBuilder {
    candidate().with_tweet_features(features)
}

fn community_post(builder: CandidateBuilder, moderation: CommunityModeration) -> PostCandidate {
    let mut post = builder
        .with_tweet_features(TweetFeatures {
            community_id: NonZeroU64::new(500),
            ..TweetFeatures::default()
        })
        .build();
    post.post.community_moderation = moderation;
    post.post.viewer_is_community_moderator = Some(false);
    post
}

fn exclusive(builder: CandidateBuilder) -> PostCandidate {
    let mut post = builder.build();
    post.post.tweet_features.exclusive_conversation_author_id = Some(42);
    post
}

fn takedown(reason: TakedownReason) -> PostCandidate {
    tweet(TweetFeatures {
        takedown_reasons: vec![reason],
        ..TweetFeatures::default()
    })
    .build()
}

fn post_cases() -> Vec<Case> {
    let filters_to_followed = || {
        recipient(
            VIEWER_ID,
            ViewerProfile {
                mention_filter_is_following: true,
                ..ViewerProfile::default()
            },
        )
    };
    let uqf = |id| {
        recipient(
            id,
            ViewerProfile {
                has_uqf_enabled: true,
                ..ViewerProfile::default()
            },
        )
    };
    let mut cases = vec![
        shown(
            "only_people_i_follow_community_post",
            "non_follower",
            filters_to_followed(),
            community_post(candidate(), CommunityModeration::default()),
        ),
        Case {
            retweet_source: allow(),
            ..dropped_at_79(
                "stale_edit",
                "non_follower",
                default_recipient(),
                tweet(TweetFeatures {
                    latest_edit_tweet_id: Some(2),
                    ..TweetFeatures::default()
                })
                .build(),
                unspecified("stale_tweet/drop/unspecified"),
            )
        },
        dropped_at_79(
            "protected_author_community_post",
            "follower",
            default_recipient(),
            community_post(
                candidate()
                    .with_author_features(AuthorFeatures {
                        is_protected: true,
                        ..AuthorFeatures::default()
                    })
                    .with_edge(Hydrator::Follows),
                CommunityModeration::default(),
            ),
            unspecified("protected_community_tweet/drop/unspecified"),
        ),
        dropped_at_79(
            "hidden_community_post",
            "non_follower",
            default_recipient(),
            community_post(
                candidate(),
                CommunityModeration {
                    is_hidden: true,
                    is_author_removed: false,
                },
            ),
            unspecified("hidden_community_tweet/drop/unspecified"),
        ),
        dropped_at_79(
            "author_removed_community_post",
            "non_follower",
            default_recipient(),
            community_post(
                candidate(),
                CommunityModeration {
                    is_hidden: false,
                    is_author_removed: true,
                },
            ),
            unspecified("author_removed_community_tweet/drop/unspecified"),
        ),
        dropped_at_79(
            "spam_label",
            "non_follower",
            default_recipient(),
            labeled(SafetyLabelType::SPAM),
            undesirable("spam/drop/undesirable"),
        ),
        shown(
            "spam_label",
            "self",
            recipient(AUTHOR_ID, ViewerProfile::default()),
            labeled(SafetyLabelType::SPAM),
        ),
        dropped_at_79(
            "pdna_label",
            "non_follower",
            default_recipient(),
            labeled(SafetyLabelType::PDNA),
            dropped(
                FilteredReason::SafetyResult(SafetyResult {
                    reason: Some(SafetyResultReason::NsfwHighPrecision),
                    action: Action::Drop(DropReason {}),
                }),
                "pdna/drop/safety_result",
            ),
        ),
        dropped_at_79(
            "bounce_label",
            "non_follower",
            default_recipient(),
            labeled(SafetyLabelType::BOUNCE),
            dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
        ),
        dropped_at_79(
            "exclusive",
            "non_follower",
            default_recipient(),
            exclusive(candidate()),
            dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
        ),
        shown(
            "exclusive_super_followed",
            "non_follower",
            default_recipient(),
            exclusive(candidate().with_edge(Hydrator::SuperFollowsExclusive)),
        ),
        dropped_at_79(
            "exclusive_super_followed_author_blocks_recipient",
            "non_follower",
            default_recipient(),
            exclusive(
                candidate()
                    .with_edge(Hydrator::SuperFollowsExclusive)
                    .with_edge(Hydrator::BlockedByAuthor),
            ),
            unspecified("author_blocks_viewer_exclusive_content/drop/unspecified"),
        ),
        dropped_at_79(
            "trusted_friends",
            "follower",
            default_recipient(),
            tweet(TweetFeatures {
                trusted_friends_list_id: Some(7),
                ..TweetFeatures::default()
            })
            .with_edge(Hydrator::Follows)
            .build(),
            unspecified("trusted_friends_tweet/drop/unspecified"),
        ),
        dropped_at_79(
            "legal_takedown_worldwide",
            "non_follower",
            default_recipient(),
            takedown(TakedownReason::LegalRequest {
                country_code: "xx".to_owned(),
            }),
            unspecified("legal_takedown/drop/unspecified"),
        ),
        dropped_at_79(
            "local_laws_takedown_de",
            "in_de",
            viewer_in_country("de"),
            takedown(TakedownReason::BystanderReport {
                country_code: "de".to_owned(),
            }),
            unspecified("local_laws_takedown/drop/unspecified"),
        ),
        dropped_at_79(
            "unpublished_article",
            "non_follower",
            default_recipient(),
            {
                let mut post = tweet(TweetFeatures {
                    article_id: NonZeroU64::new(1),
                    ..TweetFeatures::default()
                })
                .build();
                post.post.article_lifecycle = Some(ArticleLifecycle::Draft);
                post
            },
            unspecified("article_tweet_content/drop/unspecified"),
        ),
        dropped_at_79(
            "nsfw_media",
            "logged_out",
            logged_out_viewer(),
            nsfw_high_precision_media(),
            nsfw_viewer_dropped(
                NsfwViewerDropReason::LoggedOut,
                "sensitive_viewer_logged_out/drop",
            ),
        ),
        dropped_at_79(
            "nsfw_media",
            "underage",
            viewer_with_age(ViewerAge::Known(17)),
            nsfw_high_precision_media(),
            nsfw_viewer_dropped(
                NsfwViewerDropReason::IsUnderage,
                "sensitive_viewer_underage/drop",
            ),
        ),
        dropped_at_79(
            "nsfw_media",
            "no_stated_age_in_gb",
            no_stated_age_viewer("gb"),
            nsfw_high_precision_media(),
            nsfw_viewer_dropped(
                NsfwViewerDropReason::HasNoStatedAge,
                "sensitive_viewer_no_stated_age/drop",
            ),
        ),
        dropped_at_79(
            "low_quality_label",
            "non_follower",
            default_recipient(),
            labeled(SafetyLabelType::LOW_QUALITY),
            unspecified("low_quality/drop/unspecified"),
        ),
        shown(
            "low_quality_label_after_2019_04_01",
            "non_follower",
            default_recipient(),
            candidate()
                .tweet_id(AFTER_LOW_QUALITY_PROXY)
                .with_label(SafetyLabelType::LOW_QUALITY)
                .build(),
        ),
        Case {
            media_tagged: undesirable("spam_high_recall_oon/drop/undesirable"),
            ..dropped_at_79(
                "spam_high_recall_label",
                "non_follower",
                default_recipient(),
                labeled(SafetyLabelType::SPAM_HIGH_RECALL),
                undesirable("spam_high_recall/drop/undesirable"),
            )
        },
        dropped_at_79(
            "spam_high_recall_label",
            "follower",
            default_recipient(),
            candidate()
                .with_label(SafetyLabelType::SPAM_HIGH_RECALL)
                .with_edge(Hydrator::Follows)
                .build(),
            undesirable("spam_high_recall/drop/undesirable"),
        ),
        dropped_at_79(
            "duplicate_content_label",
            "non_follower",
            default_recipient(),
            labeled(SafetyLabelType::DUPLICATE_CONTENT),
            unspecified("duplicate_content/drop/unspecified"),
        ),
        dropped_at_79(
            "smyte_spam_tweet_label",
            "non_follower",
            default_recipient(),
            labeled(SafetyLabelType::SMYTE_SPAM_TWEET),
            unspecified("smyte_spam_tweet/drop/unspecified"),
        ),
        Case {
            mention_and_reply: unspecified("risky_high_viz_reply_oon/drop/unspecified"),
            media_tagged: unspecified("risky_high_viz_reply_oon/drop/unspecified"),
            ..shown(
                "risky_high_viz_reply_label",
                "non_follower",
                default_recipient(),
                labeled(SafetyLabelType::RISKY_HIGH_VIZ_REPLY),
            )
        },
        shown(
            "risky_high_viz_reply_label",
            "follower",
            default_recipient(),
            candidate()
                .with_label(SafetyLabelType::RISKY_HIGH_VIZ_REPLY)
                .with_edge(Hydrator::Follows)
                .build(),
        ),
        shown(
            "risky_high_viz_reply_label",
            "self",
            recipient(AUTHOR_ID, ViewerProfile::default()),
            labeled(SafetyLabelType::RISKY_HIGH_VIZ_REPLY),
        ),
    ];
    for (label, by) in [
        (
            SafetyLabelType::DUPLICATE_MENTION,
            "duplicate_mention_uqf/drop/unspecified",
        ),
        (SafetyLabelType::ABUSIVE, "abusive_uqf/drop/unspecified"),
        (
            SafetyLabelType::UNTRUSTED_URL,
            "untrusted_url_uqf/drop/unspecified",
        ),
        (
            SafetyLabelType::DOWNRANK_SPAM_REPLY,
            "downrank_spam_reply_uqf/drop/unspecified",
        ),
    ] {
        cases.extend([
            dropped_at_79(
                by,
                "non_follower_with_uqf",
                uqf(VIEWER_ID),
                labeled(label),
                unspecified(by),
            ),
            shown(
                by,
                "follower_with_uqf",
                uqf(VIEWER_ID),
                candidate()
                    .with_label(label)
                    .with_edge(Hydrator::Follows)
                    .build(),
            ),
            shown(
                by,
                "non_follower_without_uqf",
                default_recipient(),
                labeled(label),
            ),
        ]);
    }
    cases
}

fn post_case_verdict(kind: NotificationKind, case: &Case) -> Verdict {
    match kind {
        NotificationKind::Mention | NotificationKind::Reply => case.mention_and_reply.clone(),
        NotificationKind::MediaTagged => case.media_tagged.clone(),
        NotificationKind::Quote => allow(),
        NotificationKind::Follow
        | NotificationKind::FollowRequestAccepted
        | NotificationKind::Like
        | NotificationKind::Repost
        | NotificationKind::ContactJoined
        | NotificationKind::AddedToList => {
            unreachable!("{kind:?} is no post kind")
        }
    }
}

pub(super) fn cases() -> Vec<CorpusCase> {
    let post_kinds = SafetyLevel::VARIANTS
        .iter()
        .flat_map(|&level| level.post_kinds().map(move |(kind, _)| (level, kind)))
        .flat_map(|(level, kind)| {
            let posts = post_cases().into_iter().map(move |case| {
                let expected = post_case_verdict(kind, &case);
                CorpusCase {
                    name: format!("{}/{level:?}/{kind:?}/{}", case.name, case.role),
                    level,
                    judging: Judging::PostKind(kind),
                    viewer: case.recipient,
                    subject: Subject::Post(case.post),
                    placement: Placement::Standalone,
                    expected,
                }
            });
            post_kind_cases(level, kind).into_iter().chain(posts)
        });
    let retweet_sources = SafetyLevel::VARIANTS
        .iter()
        .filter(|level| level.retweet_source_rules().is_some())
        .flat_map(|&level| {
            post_cases().into_iter().map(move |case| CorpusCase {
                name: format!("{}/{level:?}/retweet_source/{}", case.name, case.role),
                level,
                judging: Judging::RetweetSource,
                viewer: case.recipient,
                subject: Subject::Post(case.post),
                placement: Placement::RetweetSource,
                expected: case.retweet_source,
            })
        });
    post_kinds.chain(retweet_sources).collect()
}
