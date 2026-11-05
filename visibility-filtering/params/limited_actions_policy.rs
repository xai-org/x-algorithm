use crate::metrics::families::LIMITED_ACTIONS_POLICY_MISSING;
use crate::metrics::Metrics;
use crate::models::{LimitedActionType, LimitedEngagementReason};
use crate::params::limited_actions_copy::{Prompt, PromptCopy};
use xai_feature_switches::FeatureSwitchResults;
use xai_stringcenter::DEFAULT_LANGUAGE;

pub(super) const REASON_FIELD: &str = "LimitedEngagementReason";
const LIMITED_ACTIONS_KEY: &str = "limited_actions_policy_limited_actions";
const COPY_NAMESPACE_KEY: &str = "limited_actions_policy_copy_namespace";
const PROMPT_TYPE_KEY: &str = "limited_actions_policy_prompt_type";
const LEARN_MORE_URL_KEY: &str = "limited_actions_policy_prompt_learn_more_url";

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum Missing {
    Actions,
    Copy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LimitedAction {
    pub(crate) action_type: LimitedActionType,
    pub(crate) prompt: Option<Prompt>,
}

#[derive(Default)]
pub(crate) struct LimitedActionsPolicies {
    policies: Vec<(LimitedEngagementReason, Vec<LimitedAction>)>,
    language: Option<String>,
}

impl LimitedActionsPolicies {
    pub(super) fn resolve(
        reasons: impl IntoIterator<Item = LimitedEngagementReason>,
        match_reason: impl Fn(LimitedEngagementReason) -> FeatureSwitchResults,
        prompt: impl Fn(LimitedActionType, &PromptCopy) -> Option<Prompt>,
        metrics: &Metrics,
    ) -> Self {
        let mut matched: Vec<LimitedEngagementReason> = Vec::new();
        let mut policies = Vec::new();
        let count = |reason: LimitedEngagementReason, missing: Missing| {
            metrics.count(
                &LIMITED_ACTIONS_POLICY_MISSING,
                [reason.into(), missing.into()],
                1,
            );
        };
        for reason in reasons {
            if matched.contains(&reason) {
                continue;
            }
            matched.push(reason);
            let results = match_reason(reason);
            let action_types: Vec<LimitedActionType> = results
                .get_array_no_impression(LIMITED_ACTIONS_KEY)
                .into_iter()
                .flatten()
                .filter_map(|name| name.as_str()?.parse().ok())
                .collect();
            if action_types.is_empty() {
                count(reason, Missing::Actions);
                continue;
            }
            let copy = PromptCopy::from_switches(
                results.get_string_no_impression(COPY_NAMESPACE_KEY),
                results.get_string_no_impression(PROMPT_TYPE_KEY),
                results.get_string_no_impression(LEARN_MORE_URL_KEY),
            );
            let actions: Vec<LimitedAction> = action_types
                .into_iter()
                .map(|action_type| LimitedAction {
                    action_type,
                    prompt: copy.as_ref().and_then(|copy| prompt(action_type, copy)),
                })
                .collect();
            if copy.is_some() && actions.iter().any(|action| action.prompt.is_none()) {
                count(reason, Missing::Copy);
            }
            policies.push((reason, actions));
        }
        Self {
            policies,
            language: None,
        }
    }

    pub(super) fn in_language(mut self, language: Option<&str>) -> Self {
        self.language = language.map(str::to_owned);
        self
    }

    pub(crate) fn language(&self) -> &str {
        self.language.as_deref().unwrap_or(DEFAULT_LANGUAGE)
    }

    pub(crate) fn actions(&self, reason: LimitedEngagementReason) -> &[LimitedAction] {
        self.policies
            .iter()
            .find(|(held, _)| *held == reason)
            .map_or(&[], |(_, actions)| actions)
    }

    #[cfg(test)]
    pub(crate) fn for_tests(
        policies: Vec<(LimitedEngagementReason, Vec<LimitedActionType>)>,
    ) -> Self {
        let unprompted = |action_type| LimitedAction {
            action_type,
            prompt: None,
        };
        Self {
            policies: policies
                .into_iter()
                .map(|(reason, types)| (reason, types.into_iter().map(unprompted).collect()))
                .collect(),
            language: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::limited_actions_copy::LimitedActionsCopy;
    use crate::params::ClientSwitches;
    use arc_swap::ArcSwap;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use xai_feature_switches::FeatureSwitches;
    use LimitedEngagementReason::{
        BlockedViewer, CommunityTweetCommunityDeleted, CommunityTweetCommunityNotFound,
        CommunityTweetCommunitySuspended, CommunityTweetHidden, CommunityTweetMemberRemoved,
        CommunityTweetViewerRemoved, ConversationControl, DynamicProductAd, LocalTweet,
        NonCompliant, ReadonlyViewer, RootAuthorBlockedViewer, StaleTweet,
    };

    fn copy_naming_namespaces<'a>(
        policies: impl IntoIterator<Item = (&'a str, &'a [LimitedActionType])>,
    ) -> LimitedActionsCopy {
        let mut keys = BTreeMap::new();
        for (namespace, actions) in policies {
            keys.insert(format!("{namespace}_GenericSubtext"), namespace);
            for action in actions {
                keys.insert(format!("{namespace}_{action:?}_Headline"), namespace);
                keys.insert(format!("{namespace}_{action:?}_Subtext"), namespace);
            }
        }
        let entries: Vec<String> = keys
            .iter()
            .map(|(key, namespace)| {
                format!(
                    r#"{{"string_key": "{key}", "current_variant": "{namespace}", "instructions_v2": []}}"#
                )
            })
            .collect();
        LimitedActionsCopy::from_json(&format!("[{}]", entries.join(",")))
    }

    #[test]
    fn each_reason_resolves_its_policy_from_the_scala_file() {
        use LimitedActionType as T;
        let community_unavailable = vec![
            T::AddToBookmarks,
            T::AddToMoment,
            T::Embed,
            T::Follow,
            T::HideCommunityTweet,
            T::Like,
            T::ListsAddRemove,
            T::MuteConversation,
            T::PinToProfile,
            T::QuoteTweet,
            T::React,
            T::RemoveFromCommunity,
            T::Reply,
            T::Retweet,
            T::SendViaDm,
            T::ShareTweetVia,
            T::ViewHiddenReplies,
            T::ViewTweetActivity,
            T::VoteOnPoll,
        ];
        let community_hidden = community_unavailable
            .iter()
            .copied()
            .filter(|action| *action != T::RemoveFromCommunity)
            .collect();
        let expected = [
            (
                NonCompliant,
                Some("NonCompliant"),
                vec![
                    T::Reply,
                    T::Retweet,
                    T::Like,
                    T::React,
                    T::SendViaDm,
                    T::AddToBookmarks,
                    T::AddToMoment,
                    T::PinToProfile,
                    T::ViewTweetActivity,
                    T::ShareTweetVia,
                    T::CopyLink,
                ],
            ),
            (ConversationControl, Some("LimitedReplies"), vec![T::Reply]),
            (
                BlockedViewer,
                Some("BlockedViewer"),
                vec![
                    T::Reply,
                    T::Retweet,
                    T::QuoteTweet,
                    T::Like,
                    T::React,
                    T::SendViaDm,
                    T::AddToBookmarks,
                    T::AddToMoment,
                    T::PinToProfile,
                    T::ShareTweetVia,
                    T::Follow,
                    T::ListsAddRemove,
                    T::Embed,
                    T::CopyLink,
                    T::VoteOnPoll,
                    T::ShowRetweetActionMenu,
                    T::ReplyDownVote,
                    T::EditTweet,
                    T::Highlight,
                ],
            ),
            (
                RootAuthorBlockedViewer,
                Some("BlockedViewer"),
                vec![T::Reply],
            ),
            (
                ReadonlyViewer,
                Some("ReadonlyViewer"),
                vec![
                    T::AddToBookmarks,
                    T::AddToMoment,
                    T::Embed,
                    T::HideCommunityTweet,
                    T::Like,
                    T::ListsAddRemove,
                    T::PinToProfile,
                    T::QuoteTweet,
                    T::React,
                    T::Reply,
                    T::Retweet,
                    T::VoteOnPoll,
                    T::EditTweet,
                    T::Highlight,
                ],
            ),
            (
                StaleTweet,
                None,
                vec![
                    T::Reply,
                    T::Retweet,
                    T::QuoteTweet,
                    T::Like,
                    T::React,
                    T::SendViaDm,
                    T::AddToBookmarks,
                    T::ShareTweetVia,
                    T::AddToMoment,
                    T::Embed,
                    T::HideCommunityTweet,
                    T::PinToProfile,
                    T::ViewTweetActivity,
                    T::VoteOnPoll,
                    T::CopyLink,
                ],
            ),
            (
                CommunityTweetHidden,
                Some("CommunityHidden"),
                community_hidden,
            ),
            (
                CommunityTweetMemberRemoved,
                Some("Default"),
                community_unavailable.clone(),
            ),
            (
                CommunityTweetCommunityNotFound,
                Some("Default"),
                community_unavailable.clone(),
            ),
            (
                CommunityTweetCommunityDeleted,
                Some("Default"),
                vec![T::HideCommunityTweet, T::RemoveFromCommunity],
            ),
            (
                CommunityTweetCommunitySuspended,
                Some("Default"),
                community_unavailable,
            ),
            (
                CommunityTweetViewerRemoved,
                Some("ViewerIsRemovedFromCommunity"),
                vec![T::Reply, T::PinToProfile],
            ),
            (
                DynamicProductAd,
                None,
                vec![
                    T::Reply,
                    T::Retweet,
                    T::QuoteTweet,
                    T::ShareTweetVia,
                    T::SendViaDm,
                    T::CopyLink,
                    T::AddToBookmarks,
                    T::Like,
                ],
            ),
            (LocalTweet, Some("LocalPost"), vec![T::Retweet, T::Reply]),
        ];
        let copy = copy_naming_namespaces(
            expected
                .iter()
                .filter_map(|(_, namespace, actions)| Some(((*namespace)?, actions.as_slice()))),
        );
        let (metrics, recording) = Metrics::recording();
        let policies = ClientSwitches::for_tests_with(copy).limited_actions_policies(
            None,
            None,
            None,
            expected.iter().map(|(reason, _, _)| *reason),
            &metrics,
        );
        for (reason, namespace, actions) in expected {
            let resolved = policies.actions(reason);
            let action_types: Vec<LimitedActionType> =
                resolved.iter().map(|action| action.action_type).collect();
            assert_eq!(action_types, actions, "{reason:?}");
            for action in resolved {
                assert_eq!(
                    action
                        .prompt
                        .as_ref()
                        .map(|prompt| prompt.headline.as_str()),
                    namespace,
                    "{reason:?} {:?}",
                    action.action_type
                );
            }
        }
        let counted = recording.take(&LIMITED_ACTIONS_POLICY_MISSING.family);
        assert!(counted.is_empty(), "{counted:?}");
    }

    #[test]
    fn a_reason_without_a_known_action_has_no_policy_and_counts_once() {
        const POLICY: &str = r#"
limited_actions_policy:
  parameters:
    limited_actions_policy_limited_actions: {type: array, default: []}
  rules:
  - query: "[$LimitedEngagementReason eq limited_replies]"
    values:
      limited_actions_policy_limited_actions: ["reply"]
  - query: "[$LimitedEngagementReason eq blocked_viewer]"
    values:
      limited_actions_policy_limited_actions: ["no_such_action"]
"#;
        let switches = ClientSwitches::new(
            Arc::new(ArcSwap::from_pointee(
                FeatureSwitches::load_string(POLICY).unwrap(),
            )),
            LimitedActionsCopy::from_json("[]"),
        );
        let (metrics, recording) = Metrics::recording();
        let policies = switches.limited_actions_policies(
            None,
            None,
            None,
            [StaleTweet, ConversationControl, BlockedViewer, StaleTweet],
            &metrics,
        );
        assert_eq!(
            policies.actions(ConversationControl),
            [LimitedAction {
                action_type: LimitedActionType::Reply,
                prompt: None,
            }]
        );
        assert!(policies.actions(StaleTweet).is_empty());
        assert!(policies.actions(BlockedViewer).is_empty());
        let missing = |reason: &str, missing: &str| {
            format!("limited_actions_policy_missing{{reason={reason},missing={missing}}} 1")
        };
        let counted: Vec<String> = recording
            .take(&LIMITED_ACTIONS_POLICY_MISSING.family)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            counted,
            [
                missing("stale_tweet", "actions"),
                missing("conversation_control", "copy"),
                missing("blocked_viewer", "actions"),
            ]
        );
    }
}
