use crate::hydration::{AuthorNotFound, RequestRows};
use crate::models::NotificationKind;
use crate::rules::levels;
use crate::rules::levels::{LevelRules, PostRules};
use crate::rules::notification_kind::{self, Judged};
use crate::rules::rule_spec::rule::UserRule;
use crate::rules::RuleId;
use strum::VariantArray;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum Rpc {
    FilterTweets,
    EvaluateTweets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
pub enum SafetyLevel {
    FilterAll,
    TimelineHome,
    TimelineHomeRecommendations,
    TimelineHomeHydration,
    TimelineHomePromotedHydration,
    ImmersiveExpandedRecommendations,
    QuotedTweetRules,
    NotificationsFollow,
    NotificationsWriterV2,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum WireLevel {
    FilterTweets(vf_pb::SafetyLevel),
    EvaluateTweets(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LevelError {
    Unknown,
    NotServed,
}

pub(crate) enum LevelSpec {
    Posts(PostLevel),
    QuotePass {
        rules: UserRules,
    },
    Notifications {
        thrift: ThriftLevel,
        rules: UserRules,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct UserRules(&'static [&'static [&'static UserRule]]);

pub(crate) struct PostLevel {
    proto: Option<vf_pb::SafetyLevel>,
    evaluate_tweets: Option<ThriftLevel>,
    rules: PostRules,
    unspecified_drop_is_exact: bool,
    quote_entries: QuoteEntries,
}

#[derive(Clone, Copy)]
pub(crate) enum QuoteEntries {
    NotJudged,
    Judged,
    JudgedWithQuotePass {
        rerenders: &'static [(RuleId, RerenderReason)],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RerenderReason {
    AuthorDeactivated,
    AuthorInactive,
    AuthorSuspended,
    AuthorBlocksViewer,
    AuthorProtected,
    Unspecified,
}

static TIMELINE_HOME_HYDRATION_QUOTE_RERENDERS: [(RuleId, RerenderReason); 8] = [
    (RuleId::DeactivatedAuthor, RerenderReason::AuthorDeactivated),
    (RuleId::ErasedAuthor, RerenderReason::AuthorDeactivated),
    (RuleId::OffboardedAuthor, RerenderReason::AuthorInactive),
    (RuleId::SuspendedAuthor, RerenderReason::AuthorSuspended),
    (
        RuleId::QuotedAuthorBlocksQuotingAuthor,
        RerenderReason::AuthorBlocksViewer,
    ),
    (RuleId::ViewerBlocksAuthor, RerenderReason::Unspecified),
    (
        RuleId::QuotedViewerMutesAuthorNotFollowing,
        RerenderReason::Unspecified,
    ),
    (
        RuleId::ProtectedQuotedAuthor,
        RerenderReason::AuthorProtected,
    ),
];

impl SafetyLevel {
    pub(crate) const fn spec(self) -> LevelSpec {
        match self {
            SafetyLevel::FilterAll => LevelSpec::Posts(PostLevel {
                proto: Some(vf_pb::SafetyLevel::FilterAll),
                evaluate_tweets: Some(ThriftLevel::FILTER_ALL),
                rules: levels::FILTER_ALL,
                unspecified_drop_is_exact: true,
                quote_entries: QuoteEntries::NotJudged,
            }),
            SafetyLevel::TimelineHome => LevelSpec::Posts(PostLevel {
                proto: Some(vf_pb::SafetyLevel::TimelineHome),
                evaluate_tweets: Some(ThriftLevel::TIMELINE_HOME),
                rules: levels::TIMELINE_HOME,
                unspecified_drop_is_exact: false,
                quote_entries: QuoteEntries::NotJudged,
            }),
            SafetyLevel::TimelineHomeRecommendations => LevelSpec::Posts(PostLevel {
                proto: Some(vf_pb::SafetyLevel::TimelineHomeRecommendations),
                evaluate_tweets: Some(ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS),
                rules: levels::TIMELINE_HOME_RECOMMENDATIONS,
                unspecified_drop_is_exact: false,
                quote_entries: QuoteEntries::NotJudged,
            }),
            SafetyLevel::TimelineHomeHydration => LevelSpec::Posts(PostLevel {
                proto: None,
                evaluate_tweets: Some(ThriftLevel::TIMELINE_HOME_HYDRATION),
                rules: levels::TIMELINE_HOME_HYDRATION,
                unspecified_drop_is_exact: false,
                quote_entries: QuoteEntries::JudgedWithQuotePass {
                    rerenders: &TIMELINE_HOME_HYDRATION_QUOTE_RERENDERS,
                },
            }),
            SafetyLevel::TimelineHomePromotedHydration => LevelSpec::Posts(PostLevel {
                proto: None,
                evaluate_tweets: Some(ThriftLevel::TIMELINE_HOME_PROMOTED_HYDRATION),
                rules: levels::TIMELINE_HOME_PROMOTED_HYDRATION,
                unspecified_drop_is_exact: false,
                quote_entries: QuoteEntries::Judged,
            }),
            SafetyLevel::ImmersiveExpandedRecommendations => LevelSpec::Posts(PostLevel {
                proto: Some(vf_pb::SafetyLevel::ImmersiveExpandedRecommendations),
                evaluate_tweets: None,
                rules: levels::IMMERSIVE_EXPANDED_RECOMMENDATIONS,
                unspecified_drop_is_exact: false,
                quote_entries: QuoteEntries::NotJudged,
            }),
            SafetyLevel::QuotedTweetRules => LevelSpec::QuotePass {
                rules: UserRules(levels::QUOTED_TWEET_RULES),
            },
            SafetyLevel::NotificationsFollow => LevelSpec::Notifications {
                thrift: ThriftLevel::NOTIFICATIONS_FOLLOW,
                rules: UserRules(levels::NOTIFICATIONS_FOLLOW),
            },
            SafetyLevel::NotificationsWriterV2 => LevelSpec::Notifications {
                thrift: ThriftLevel::NOTIFICATIONS_WRITER_V2,
                rules: UserRules(levels::NOTIFICATIONS_WRITER_V2),
            },
        }
    }

    pub(crate) fn parse(wire: WireLevel) -> Result<Self, LevelError> {
        match wire {
            WireLevel::FilterTweets(proto) => Self::served_where(|spec| match spec {
                LevelSpec::Posts(post) => post.proto == Some(proto),
                LevelSpec::QuotePass { .. } | LevelSpec::Notifications { .. } => false,
            }),
            WireLevel::EvaluateTweets(number) => {
                let number = known_thrift(number)?;
                Self::served_where(|spec| match spec {
                    LevelSpec::Posts(post) => post.evaluate_tweets == Some(number),
                    LevelSpec::QuotePass { .. } | LevelSpec::Notifications { .. } => false,
                })
            }
        }
    }

    pub(crate) fn notification_level(number: i32) -> Result<Self, LevelError> {
        let number = known_thrift(number)?;
        Self::served_where(|spec| match spec {
            LevelSpec::Notifications { thrift, .. } => *thrift == number,
            LevelSpec::Posts(_) | LevelSpec::QuotePass { .. } => false,
        })
    }

    pub(crate) fn judges(self, kind: NotificationKind) -> bool {
        matches!(self.spec(), LevelSpec::Notifications { thrift, .. } if thrift == kind.level())
    }

    pub(super) fn post_kinds(self) -> impl Iterator<Item = (NotificationKind, PostRules)> {
        NotificationKind::VARIANTS
            .iter()
            .copied()
            .filter(move |&kind| self.judges(kind))
            .filter_map(|kind| match notification_kind::judged(kind) {
                Judged::ActorPost(rules) => Some((kind, rules)),
                Judged::Actor => None,
            })
    }

    #[cfg(test)]
    pub(super) fn rule_lists(self) -> impl Iterator<Item = levels::LevelLists> {
        std::iter::once(self.spec().rules().lists())
            .chain(self.post_kinds().map(|(_, rules)| rules.lists()))
    }

    fn served_where(serves: impl Fn(&LevelSpec) -> bool) -> Result<Self, LevelError> {
        Self::VARIANTS
            .iter()
            .copied()
            .find(|variant| serves(&variant.spec()))
            .ok_or(LevelError::NotServed)
    }

    pub(crate) fn may_merge_retweets(self) -> bool {
        match self.spec() {
            LevelSpec::Posts(post) => post.evaluate_tweets.is_some(),
            LevelSpec::QuotePass { .. } | LevelSpec::Notifications { .. } => false,
        }
    }

    pub(crate) fn is_carried_by(self, rpc: Rpc) -> bool {
        match (rpc, self.spec()) {
            (Rpc::FilterTweets, LevelSpec::Posts(post)) => post.proto.is_some(),
            (Rpc::FilterTweets, LevelSpec::QuotePass { .. } | LevelSpec::Notifications { .. }) => {
                false
            }
            (Rpc::EvaluateTweets, _) => self.may_merge_retweets(),
        }
    }
}

impl LevelSpec {
    pub(super) const fn rules(&self) -> LevelRules {
        match self {
            Self::Posts(post) => LevelRules::UserThenPost(post.rules),
            Self::QuotePass {
                rules: UserRules(rules),
            }
            | Self::Notifications {
                rules: UserRules(rules),
                ..
            } => LevelRules::UserOnly(rules),
        }
    }

    pub(crate) const fn rows(&self) -> RequestRows {
        match self {
            Self::Posts(_) | Self::QuotePass { .. } => RequestRows::Posts,
            Self::Notifications { .. } => RequestRows::Users,
        }
    }

    pub(crate) const fn author_not_found(&self) -> AuthorNotFound {
        match self {
            Self::Posts(_) | Self::Notifications { .. } => AuthorNotFound::Unresolves,
            Self::QuotePass { .. } => AuthorNotFound::ReadsErased,
        }
    }

    pub(crate) const fn unspecified_drop_is_exact(&self) -> bool {
        match self {
            Self::Posts(post) => post.unspecified_drop_is_exact,
            Self::QuotePass { .. } | Self::Notifications { .. } => false,
        }
    }

    pub(crate) const fn quote_entries(&self) -> QuoteEntries {
        match self {
            Self::Posts(post) => post.quote_entries,
            Self::QuotePass { .. } | Self::Notifications { .. } => QuoteEntries::NotJudged,
        }
    }
}

fn known_thrift(number: i32) -> Result<ThriftLevel, LevelError> {
    let thrift = ThriftLevel(number);
    if ThriftLevel::ENUM_VALUES.contains(&thrift) {
        Ok(thrift)
    } else {
        Err(LevelError::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_filter_tweets_level_reaches_its_level() {
        let reached: Vec<_> = (0..=255)
            .filter_map(|number| vf_pb::SafetyLevel::try_from(number).ok())
            .map(|proto| {
                (
                    proto,
                    SafetyLevel::parse(WireLevel::FilterTweets(proto)).ok(),
                )
            })
            .collect();
        assert_eq!(
            reached,
            [
                (vf_pb::SafetyLevel::FilterAll, Some(SafetyLevel::FilterAll)),
                (
                    vf_pb::SafetyLevel::TimelineHome,
                    Some(SafetyLevel::TimelineHome)
                ),
                (
                    vf_pb::SafetyLevel::TimelineHomeRecommendations,
                    Some(SafetyLevel::TimelineHomeRecommendations)
                ),
                (
                    vf_pb::SafetyLevel::ImmersiveExpandedRecommendations,
                    Some(SafetyLevel::ImmersiveExpandedRecommendations)
                ),
            ]
        );
    }

    #[test]
    fn evaluate_tweets_serves_these_thrift_levels() {
        let served: Vec<_> = ThriftLevel::ENUM_VALUES
            .iter()
            .filter_map(|&thrift| {
                SafetyLevel::parse(WireLevel::EvaluateTweets(thrift.0))
                    .ok()
                    .map(|level| (thrift, level))
            })
            .collect();
        assert_eq!(
            served,
            [
                (ThriftLevel::TIMELINE_HOME, SafetyLevel::TimelineHome),
                (ThriftLevel::FILTER_ALL, SafetyLevel::FilterAll),
                (
                    ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS,
                    SafetyLevel::TimelineHomeRecommendations
                ),
                (
                    ThriftLevel::TIMELINE_HOME_HYDRATION,
                    SafetyLevel::TimelineHomeHydration
                ),
                (
                    ThriftLevel::TIMELINE_HOME_PROMOTED_HYDRATION,
                    SafetyLevel::TimelineHomePromotedHydration
                ),
            ]
        );
    }

    fn wire_ids(spec: &LevelSpec) -> (Option<vf_pb::SafetyLevel>, Option<i32>) {
        match spec {
            LevelSpec::Posts(post) => (post.proto, post.evaluate_tweets.map(|thrift| thrift.0)),
            LevelSpec::QuotePass { .. } => (None, None),
            LevelSpec::Notifications { thrift, .. } => (None, Some(thrift.0)),
        }
    }

    #[test]
    fn every_served_level_takes_the_number_the_thrift_enum_gives_it() {
        let numbers: Vec<_> = SafetyLevel::VARIANTS
            .iter()
            .map(|level| wire_ids(&level.spec()).1)
            .collect();
        assert_eq!(
            numbers,
            [
                Some(16),
                Some(8),
                Some(61),
                Some(82),
                Some(124),
                None,
                None,
                Some(246),
                Some(72)
            ]
        );
    }

    #[test]
    fn each_level_has_rules_an_ingress_and_wire_ids_of_its_own() {
        for &level in SafetyLevel::VARIANTS {
            let spec = level.spec();
            let lists = spec.rules().lists();
            assert!(
                !lists.user.is_empty() || !lists.post.is_empty(),
                "{level:?} wires no rule"
            );
            let (proto, thrift) = wire_ids(&spec);
            if let LevelSpec::Posts(_) = spec {
                assert!(
                    proto.is_some() || thrift.is_some(),
                    "{level:?} has no ingress"
                );
            }
            for &other in SafetyLevel::VARIANTS
                .iter()
                .filter(|&&other| other != level)
            {
                let (other_proto, other_thrift) = wire_ids(&other.spec());
                assert!(
                    proto.is_none() || proto != other_proto,
                    "{level:?} and {other:?} share a proto level"
                );
                assert!(
                    thrift.is_none() || thrift != other_thrift,
                    "{level:?} and {other:?} share a Thrift level"
                );
            }
        }
    }

    #[test]
    fn each_level_that_judges_quote_entries_rerenders_every_quoted_tweet_rule() {
        let mut rules: Vec<&str> = crate::rules::RuleEngine::for_tests()
            .wired_rule_names(SafetyLevel::QuotedTweetRules)
            .into_iter()
            .map(|name| RuleId::of_rule_name(name).unwrap().into())
            .collect();
        rules.sort_unstable();
        for &level in SafetyLevel::VARIANTS {
            let QuoteEntries::JudgedWithQuotePass { rerenders } = level.spec().quote_entries()
            else {
                continue;
            };
            let mut rows: Vec<&str> = rerenders.iter().map(|&(rule, _)| rule.into()).collect();
            rows.sort_unstable();
            assert_eq!(rows, rules, "{level:?}");
        }
    }
}
