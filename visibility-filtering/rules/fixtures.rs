use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    AuthorFeatures, AuthorLabel, ClientCapability, ClientSwitch, ClientSwitchSet,
    ConversationControlFeatures, Decided, DropReason, EdgeSet, FosnrReason, LabelInterstitial,
    LimitedActionType, LimitedEngagement, LimitedEngagementReason, MediaInterstitial,
    MediaRestriction, Notice, NsfwViewerDropReason, PostCandidate, PostEntity, PublicInterest,
    RequestFacts, SafetyLabelMap, SafetyLabelType, TombstoneReason, TweetFeatures, UserEntity,
    Verdict, VerifyBlurSupport, Viewer, ViewerAge, ViewerFeatures, ViewerProfile, Withholding,
};
use crate::rules::RuleId;
use std::collections::HashSet;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{AppealablePolicy, InterstitialAction, InterstitialReason};

const TWEET_ID: u64 = 1;
pub(super) const AUTHOR_ID: u64 = 100;
pub(crate) const VIEWER_ID: u64 = 999;

#[derive(Clone, Copy)]
pub(crate) struct ClientClass {
    pub(crate) name: &'static str,
    pub(crate) app_id: i64,
    pub(crate) user_agent: &'static str,
    pub(crate) capability: ClientCapability,
}

pub(crate) const CLIENT_CLASSES: [ClientClass; 8] = {
    use ClientSwitch::{
        CommunityViewerRemovedLimits, FosnrFallbackDrops, FosnrRules, GoreBlurIgnoresSettings,
        ModernBlur,
    };
    use VerifyBlurSupport::{AndroidNeedsUpdate, IosNeedsUpdate, Supported, Unsupported};
    const RWEB: i64 = 3033300;
    const IPHONE: i64 = 129032;
    const ANDROID: i64 = 258901;
    const MAC: i64 = 557701;
    const LEGACY: ClientSwitchSet = ClientSwitchSet::DEFAULT_ON;
    const MODERN: ClientSwitchSet = LEGACY.with(ModernBlur).with(GoreBlurIgnoresSettings);
    const fn class(
        name: &'static str,
        app_id: i64,
        user_agent: &'static str,
        verify_blur_support: VerifyBlurSupport,
        switches_on: ClientSwitchSet,
    ) -> ClientClass {
        let mut capability = ClientCapability {
            verify_blur_support: Some(verify_blur_support),
            switches_on,
        };
        if matches!(app_id, RWEB | IPHONE | ANDROID) {
            capability = capability.with(CommunityViewerRemovedLimits);
        }
        ClientClass {
            name,
            app_id,
            user_agent,
            capability,
        }
    }
    const fn android_9_82(class: ClientClass) -> ClientClass {
        ClientClass {
            capability: class
                .capability
                .without(FosnrRules)
                .with(FosnrFallbackDrops)
                .without(CommunityViewerRemovedLimits),
            ..class
        }
    }
    [
        class(
            "web",
            RWEB,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            Supported,
            MODERN,
        ),
        class(
            "ios_current",
            IPHONE,
            "Twitter-iPhone/11.11.5 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)",
            Supported,
            MODERN,
        ),
        class(
            "ios_outdated",
            IPHONE,
            "Twitter-iPhone/11.11.4 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)",
            IosNeedsUpdate,
            MODERN,
        ),
        class(
            "android_current",
            ANDROID,
            "TwitterAndroid/11.11.0-release.00 (311110000-r-0) Pixel 7/14 \
             (Google;panther;google;panther;0;;1;2022)",
            Supported,
            MODERN,
        ),
        class(
            "android_outdated",
            ANDROID,
            "TwitterAndroid/11.10.9-release.00 (311109000-r-0) Pixel 7/14 \
             (Google;panther;google;panther;0;;1;2022)",
            AndroidNeedsUpdate,
            MODERN,
        ),
        android_9_82(class(
            "android_without_fosnr",
            ANDROID,
            "TwitterAndroid/9.82.0-release.00 (29820000-r-0) Pixel 7/14 \
             (Google;panther;google;panther;0;;1;2022)",
            AndroidNeedsUpdate,
            LEGACY,
        )),
        class(
            "mac_app",
            MAC,
            "Twitter-Mac/11.11.5 macOS/14.0 (Apple;Mac14,2)",
            Unsupported,
            LEGACY,
        ),
        class(
            "third_party_client",
            0,
            "ThirdPartyClient/2.0",
            Unsupported,
            LEGACY,
        ),
    ]
};

fn id_of(by: &str) -> RuleId {
    RuleId::of_rule_name(by).unwrap_or(RuleId::FilterAll)
}

pub(crate) fn allow() -> Verdict {
    Verdict::Shown {
        label_interstitial: None,
        media: None,
        engagement: None,
    }
}

pub(crate) fn label_interstitial(
    interstitial: LabelInterstitial,
    by: &'static str,
    engagement: Option<(LimitedEngagementReason, &'static str)>,
) -> Verdict {
    Verdict::Shown {
        label_interstitial: Some(Decided {
            value: interstitial,
            id: id_of(by),
            by,
        }),
        media: None,
        engagement: engagement.map(|(reason, by)| Decided {
            value: LimitedEngagement::new(reason),
            id: id_of(by),
            by,
        }),
    }
}

pub(crate) fn with_label_interstitial(
    shown: Verdict,
    interstitial: LabelInterstitial,
    by: &'static str,
) -> Verdict {
    let Verdict::Shown {
        media, engagement, ..
    } = shown
    else {
        panic!("a label interstitial joins a shown verdict");
    };
    Verdict::Shown {
        label_interstitial: Some(Decided {
            value: interstitial,
            id: id_of(by),
            by,
        }),
        media,
        engagement,
    }
}

pub(crate) fn fosnr(proactive: bool, appeal_submitted: bool, by: &'static str) -> Verdict {
    Verdict::Fosnr(Decided {
        value: Notice::SoftIntervention(FosnrReason {
            policy: AppealablePolicy::ABUSE,
            level: 1,
            proactive,
            appeal_submitted,
        }),
        id: id_of(by),
        by,
    })
}

pub(crate) fn appealed(
    policy: AppealablePolicy,
    level: i8,
    proactive: bool,
    appeal_submitted: bool,
    by: &'static str,
) -> Verdict {
    use LimitedActionType as A;
    let limited_actions: &'static [LimitedActionType] = if level == 1 {
        &[A::EditTweet]
    } else {
        &[
            A::Like,
            A::Reply,
            A::Retweet,
            A::QuoteTweet,
            A::ShareTweetVia,
            A::AddToBookmarks,
            A::PinToProfile,
            A::CopyLink,
            A::SendViaDm,
            A::EditTweet,
            A::Highlight,
            A::Embed,
            A::ListsAddRemove,
        ]
    };
    Verdict::Fosnr(Decided {
        value: Notice::Appealable {
            reason: FosnrReason {
                policy,
                level,
                proactive,
                appeal_submitted,
            },
            limited_actions,
        },
        id: id_of(by),
        by,
    })
}

pub(crate) fn dropped(reason: FilteredReason, by: &'static str) -> Verdict {
    Verdict::Withheld(Decided {
        value: Withholding::Drop(DropReason::Legacy(reason)),
        id: id_of(by),
        by,
    })
}

pub(crate) fn nsfw_viewer_dropped(reason: NsfwViewerDropReason, by: &'static str) -> Verdict {
    Verdict::Withheld(Decided {
        value: Withholding::Drop(DropReason::NsfwViewer(reason)),
        id: id_of(by),
        by,
    })
}

pub(crate) fn tombstoned(reason: TombstoneReason, by: &'static str) -> Verdict {
    Verdict::Withheld(Decided {
        value: Withholding::Tombstone(reason),
        id: id_of(by),
        by,
    })
}

pub(crate) fn blurred(reason: InterstitialReason, by: &'static str) -> Verdict {
    media_blurred(reason, None, by)
}

pub(crate) fn verify_blurred(reason: InterstitialReason, by: &'static str) -> Verdict {
    media_blurred(
        reason,
        Some(InterstitialAction::AGE_VERIFICATION_PROMPT),
        by,
    )
}

fn media_blurred(
    reason: InterstitialReason,
    prompt: Option<InterstitialAction>,
    by: &'static str,
) -> Verdict {
    Verdict::Shown {
        label_interstitial: None,
        media: Some(Decided {
            value: MediaRestriction::MediaInterstitial(MediaInterstitial {
                legacy: FilteredReason::ContainNsfwMedia,
                reason,
                prompt,
            }),
            id: id_of(by),
            by,
        }),
        engagement: None,
    }
}

pub(crate) fn legacy_interstitial(by: &'static str) -> Verdict {
    Verdict::Shown {
        label_interstitial: None,
        media: Some(Decided {
            value: MediaRestriction::NsfwInterstitial,
            id: id_of(by),
            by,
        }),
        engagement: None,
    }
}

pub(crate) fn limited(reason: LimitedEngagementReason, by: &'static str) -> Verdict {
    limited_for(&[reason], by)
}

pub(crate) fn limited_for(reasons: &[LimitedEngagementReason], by: &'static str) -> Verdict {
    let (&first, rest) = reasons.split_first().unwrap();
    let mut value = LimitedEngagement::new(first);
    for &reason in rest {
        value.add(reason);
    }
    Verdict::Shown {
        label_interstitial: None,
        media: None,
        engagement: Some(Decided {
            value,
            id: id_of(by),
            by,
        }),
    }
}

pub(crate) fn builder_call(verdict: &Verdict) -> String {
    let built = match verdict {
        Verdict::Withheld(Decided { value, by, .. }) => match value {
            Withholding::Drop(DropReason::Legacy(reason)) => {
                Some(format!("dropped(FilteredReason::{reason:?}, {by:?})"))
            }
            Withholding::Drop(DropReason::NsfwViewer(reason)) => Some(format!(
                "nsfw_viewer_dropped(NsfwViewerDropReason::{reason:?}, {by:?})"
            )),
            Withholding::Tombstone(reason) => {
                Some(format!("tombstoned(TombstoneReason::{reason:?}, {by:?})"))
            }
        },
        Verdict::Fosnr(Decided { value, by, .. }) => {
            let (call, rebuilt) = match value {
                Notice::SoftIntervention(reason) => (
                    format!(
                        "fosnr({}, {}, {by:?})",
                        reason.proactive, reason.appeal_submitted
                    ),
                    fosnr(reason.proactive, reason.appeal_submitted, by),
                ),
                Notice::Appealable { reason, .. } => (
                    format!(
                        "appealed({:?}, {}, {}, {}, {by:?})",
                        reason.policy, reason.level, reason.proactive, reason.appeal_submitted
                    ),
                    appealed(
                        reason.policy,
                        reason.level,
                        reason.proactive,
                        reason.appeal_submitted,
                        by,
                    ),
                ),
            };
            (rebuilt == *verdict).then_some(call)
        }
        Verdict::Shown {
            label_interstitial: None,
            media,
            engagement,
        } => Some(slots_call(media.as_ref(), engagement.as_ref())),
        Verdict::Shown {
            label_interstitial: Some(Decided { value, by, .. }),
            media,
            engagement,
        } => pasteable_interstitial(value).map(|interstitial| match (media, engagement) {
            (None, None) => format!("label_interstitial({interstitial}, {by:?}, None)"),
            (media, engagement) => format!(
                "with_label_interstitial({}, {interstitial}, {by:?})",
                slots_call(media.as_ref(), engagement.as_ref())
            ),
        }),
    };
    built.unwrap_or_else(|| format!("{verdict:?}"))
}

fn slots_call(
    media: Option<&Decided<MediaRestriction>>,
    engagement: Option<&Decided<LimitedEngagement>>,
) -> String {
    let media_call = media.map(|Decided { value, by, .. }| match value {
        MediaRestriction::NsfwInterstitial => format!("legacy_interstitial({by:?})"),
        MediaRestriction::MediaInterstitial(blur) => {
            let call = if blur.prompt.is_some() {
                "verify_blurred"
            } else {
                "blurred"
            };
            format!("{call}(InterstitialReason::{:?}, {by:?})", blur.reason)
        }
    });
    let limit_call = engagement.map(|Decided { value, by, .. }| {
        let reasons: Vec<String> = value
            .reasons()
            .map(|reason| format!("LimitedEngagementReason::{reason:?}"))
            .collect();
        format!("limited_for(&[{}], {by:?})", reasons.join(", "))
    });
    match (media_call, limit_call) {
        (None, None) => "allow()".to_owned(),
        (Some(call), None) | (None, Some(call)) => call,
        (Some(blur), Some(limit)) => format!("blurred_and_limited({blur}, {limit})"),
    }
}

fn pasteable_interstitial(interstitial: &LabelInterstitial) -> Option<String> {
    match interstitial {
        LabelInterstitial::PublicInterest(PublicInterest { reason, limit }) => Some(format!(
            "LabelInterstitial::PublicInterest(PublicInterest {{ reason: {reason:?}, limit: {} }})",
            limit.map_or_else(
                || "None".to_owned(),
                |limit| format!("Some(LimitedEngagementReason::{limit:?})")
            )
        )),
        LabelInterstitial::EmergencyDynamic(_) => None,
    }
}

pub(crate) fn blurred_and_limited(blur: Verdict, limit: Verdict) -> Verdict {
    match (blur, limit) {
        (Verdict::Shown { media, .. }, Verdict::Shown { engagement, .. }) => Verdict::Shown {
            label_interstitial: None,
            media,
            engagement,
        },
        (blur, limit) => panic!("expected two Shown verdicts, got {blur:?} and {limit:?}"),
    }
}

pub(crate) fn viewer(id: u64) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id,
            profile: ViewerProfile::default(),
            has_age_verified_18_label: false,
        },
        ..Default::default()
    }
}

pub(crate) fn viewer_with_profile(profile: ViewerProfile) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id: VIEWER_ID,
            profile,
            has_age_verified_18_label: false,
        },
        ..Default::default()
    }
}

pub(crate) fn author_viewer() -> ViewerFeatures {
    viewer(AUTHOR_ID)
}

pub(crate) fn logged_out_viewer() -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedOut,
        ..Default::default()
    }
}

pub(crate) fn sensitive_opt_in_viewer() -> ViewerFeatures {
    viewer_with_profile(ViewerProfile {
        allows_sensitive_media: true,
        ..ViewerProfile::default()
    })
}

pub(super) fn conversation_control(
    arm: ConversationControlArm,
    root_author_id: u64,
) -> ConversationControlFeatures {
    ConversationControlFeatures {
        control: ConversationControl {
            arm,
            conversation_tweet_author_id: root_author_id,
            invited_user_ids: vec![],
            invite_via_mention: None,
            allowed_country_codes: vec![],
        },
        viewer_country: None,
    }
}

pub(super) fn author_candidate(set: fn(&mut AuthorFeatures)) -> PostCandidate {
    let mut features = AuthorFeatures::default();
    set(&mut features);
    candidate().with_author_features(features).build()
}

pub(super) fn tweet_candidate(set: fn(&mut TweetFeatures)) -> PostCandidate {
    let mut features = TweetFeatures::default();
    set(&mut features);
    candidate().with_tweet_features(features).build()
}

pub(super) fn labeled(label: SafetyLabelType) -> PostCandidate {
    candidate().with_label(label).build()
}

pub(super) fn controlled_root(arm: ConversationControlArm) -> ConversationControlFeatures {
    conversation_control(arm, AUTHOR_ID)
}

pub(super) fn viewer_in_country(code: &str) -> ViewerFeatures {
    ViewerFeatures {
        facts: RequestFacts::default().with_country(code),
        ..viewer(VIEWER_ID)
    }
}

pub(super) fn on_client(class_name: &str, country: &str, base: ViewerFeatures) -> ViewerFeatures {
    let class = CLIENT_CLASSES
        .iter()
        .find(|class| class.name == class_name)
        .unwrap_or_else(|| panic!("no client class {class_name}"));
    ViewerFeatures {
        facts: base
            .facts
            .with_country(country)
            .with_client(class.capability),
        ..base
    }
}

pub(super) fn nsfw_high_precision_media() -> PostCandidate {
    candidate()
        .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
        .with_media()
        .build()
}

pub(super) fn viewer_with_age(age: ViewerAge) -> ViewerFeatures {
    viewer_with_profile(ViewerProfile {
        viewer_age: age,
        ..ViewerProfile::default()
    })
}

pub(super) fn no_stated_age_viewer(account_country_code: &str) -> ViewerFeatures {
    viewer_with_profile(ViewerProfile {
        viewer_age: ViewerAge::NotStated,
        account_country_code: Some(account_country_code.to_string()),
        ..ViewerProfile::default()
    })
}

pub(super) fn read_only_viewer(id: u64) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id,
            profile: ViewerProfile {
                is_read_only: true,
                ..ViewerProfile::default()
            },
            has_age_verified_18_label: false,
        },
        ..ViewerFeatures::default()
    }
}

pub(crate) fn candidate() -> CandidateBuilder {
    CandidateBuilder {
        candidate: PostCandidate {
            author: UserEntity {
                id: AUTHOR_ID,
                ..Default::default()
            },
            post: PostEntity {
                tweet_id: TWEET_ID,
                ..Default::default()
            },
            edges: EdgeSet::default(),
            failed: Hydrators::empty(),
        },
        labels: HashSet::new(),
        agent_labels: HashSet::new(),
        label_sources: Vec::new(),
    }
}

pub(crate) struct CandidateBuilder {
    candidate: PostCandidate,
    labels: HashSet<SafetyLabelType>,
    agent_labels: HashSet<SafetyLabelType>,
    label_sources: Vec<(SafetyLabelType, &'static str)>,
}

impl CandidateBuilder {
    pub(crate) fn tweet_id(mut self, id: u64) -> Self {
        self.candidate.post.tweet_id = id;
        self
    }

    pub(crate) fn with_label(mut self, label: SafetyLabelType) -> Self {
        self.labels.insert(label);
        self
    }

    pub(crate) fn with_label_source(
        mut self,
        label: SafetyLabelType,
        source: &'static str,
    ) -> Self {
        self.labels.insert(label);
        self.label_sources.push((label, source));
        self
    }

    pub(crate) fn with_agent_label(mut self, label: SafetyLabelType) -> Self {
        self.agent_labels.insert(label);
        self.with_label(label)
    }

    pub(crate) fn with_author_user_label(mut self, label: AuthorLabel) -> Self {
        self.candidate.author.labels.insert(label);
        self
    }

    pub(crate) fn with_tweet_features(mut self, features: TweetFeatures) -> Self {
        self.candidate.post.tweet_features = features;
        self
    }

    pub(crate) fn with_author_features(mut self, features: AuthorFeatures) -> Self {
        self.candidate.author = UserEntity {
            id: self.candidate.author.id,
            features,
            labels: self.candidate.author.labels,
            ..UserEntity::default()
        };
        self
    }

    pub(crate) fn with_edge(mut self, edge: Hydrator) -> Self {
        debug_assert!(edge.is_viewer_edge(), "{edge:?} is not a viewer edge node");
        debug_assert!(
            !Hydrators::BLOCKED_BY_CONVERSATION_ROOT_AUTHOR.contains(edge),
            "{edge:?} is read through root_author_blocks_viewer()"
        );
        self.candidate.edges = self.candidate.edges.with(edge);
        self
    }

    pub(crate) fn with_media(mut self) -> Self {
        self.candidate.post.tweet_features.media.has_media = true;
        self
    }

    pub(crate) fn root_author_blocks_viewer(mut self) -> Self {
        self.candidate.post.conversation_root_author_blocks_viewer = true;
        self
    }

    pub(crate) fn with_conversation_control(
        mut self,
        features: ConversationControlFeatures,
    ) -> Self {
        self.candidate.post.conversation_control = Some(features);
        self
    }

    pub(crate) fn retweet_of(mut self, source_tweet_id: u64) -> Self {
        self.candidate.post.source_tweet_id = Some(source_tweet_id);
        self
    }

    pub(crate) fn with_reacted_to_post_labels(mut self, labels: &[SafetyLabelType]) -> Self {
        self.candidate.post.reacted_to_post_safety_labels =
            SafetyLabelMap::new(labels.iter().copied().collect());
        self
    }

    pub(crate) fn with_protected_reacted_to_author(mut self) -> Self {
        self.candidate.post.reacted_to_author_is_protected = true;
        self
    }

    pub(crate) fn build(self) -> PostCandidate {
        let mut candidate = self.candidate;
        if !self.labels.is_empty() {
            let labels = self.label_sources.into_iter().fold(
                SafetyLabelMap::new(self.labels),
                |labels, (label, source)| labels.with_source(label, source),
            );
            candidate.post.safety_labels = self
                .agent_labels
                .into_iter()
                .fold(labels, SafetyLabelMap::assigned_by_agent);
        }
        candidate
    }
}
