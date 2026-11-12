use crate::models::Evaluation;
use crate::params::LimitedActionsPolicies;
use crate::quote::QuoteRendering;
use crate::rules::SafetyLevel;
use crate::treatment;
use std::fmt::Debug;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::action::Action;
use xai_x_thrift::get_tweet_fields::TweetFieldsResultState;
use xai_x_thrift::safety_result::FilteredReason;

pub(crate) const NONE: &str = "none";

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Class {
    Allow,
    BareDrop,
    Drop,
    Suppressed,
    NotFound,
    Failed,
    Missing,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Failure {
    NoValue,
    StratoError,
    RateLimited,
    DecodeError,
    RpcError,
    OverCapacity,
    TweetypieFailed,
}

pub(crate) type Label = (String, String);

const SCREENED_DROPS: [&str; 9] = [
    "tweet_is_bounced",
    "author_block_viewer",
    "author_is_protected",
    "author_is_suspended",
    "exclusive_tweet",
    "nsfw_viewer_is_underage",
    "nsfw_viewer_has_no_stated_age",
    "nsfw_logged_out",
    "premium_tweet",
];

pub(crate) fn renders_alike(vf: &Label, tweetypie: &Label) -> bool {
    vf == tweetypie
        || (vf.0 == <&str>::from(Class::BareDrop)
            && tweetypie.0 == <&str>::from(Class::Drop)
            && !SCREENED_DROPS.contains(&tweetypie.1.as_str()))
}

pub(crate) fn label(class: Class, reason: impl Into<String>) -> Label {
    (<&str>::from(class).to_string(), reason.into())
}

pub(crate) fn failed(failure: Failure) -> Label {
    label(Class::Failed, <&str>::from(failure))
}

fn ident(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(s.len());
    s.split_at(end)
}

fn snake(ident: &str) -> String {
    let mut out = String::with_capacity(ident.len() + 4);
    for (i, c) in ident.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

pub(crate) fn debug_label(value: &impl Debug) -> Label {
    let debug = format!("{value:?}");
    let class = snake(ident(&debug).0);
    let Some((_, mut rest)) = debug.split_once("reason: Some(") else {
        return (class, NONE.to_string());
    };
    let reason = loop {
        let (name, after) = ident(rest);
        if let Some(inner) = after.strip_prefix(" {")
            && let Some((_, nested)) = inner.split_once("Some(")
        {
            rest = nested;
            continue;
        }
        let number = after
            .strip_prefix('(')
            .and_then(|r| r.split(')').next())
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit() || c == '-'));
        break match (name, number) {
            ("", _) => "unknown".to_string(),
            (name, Some(n)) => format!("{}_{n}", snake(name)),
            (name, None) => snake(name),
        };
    };
    (class, reason)
}

pub(crate) fn quote_label(rendering: QuoteRendering<'_>, served: Label) -> Label {
    match rendering {
        QuoteRendering::Served => served,
        QuoteRendering::Rerendered(rerendered) => {
            state_label(&treatment::rerendered_state(rerendered.reason))
        }
        QuoteRendering::Omitted => label(Class::Missing, NONE),
    }
}

pub(crate) fn filter_label(result: &vf_pb::TweetVisibilityResult) -> Label {
    let kind = result.action.and_then(|action| action.kind);
    let reason = result
        .filtered_reason
        .as_ref()
        .and_then(|r| r.reason.as_ref());
    (
        kind.map_or_else(|| NONE.to_string(), |kind| debug_label(&kind).0),
        reason.map_or_else(|| NONE.to_string(), |reason| debug_label(reason).0),
    )
}

pub(crate) fn evaluate_tweets_label(
    evaluation: &Evaluation,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Label {
    treatment::evaluate_tweets_state(evaluation, level, policies)
        .map_or_else(|| label(Class::Failed, NONE), |state| state_label(&state))
}

pub(crate) fn state_label(state: &TweetFieldsResultState) -> Label {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "FilteredReason is a generated Thrift union; every variant but SafetyResult labels by its Debug name, including any the IDL adds"
    )]
    let reason = |reason: &FilteredReason| match reason {
        FilteredReason::SafetyResult(result) => debug_label(result).1,
        other => debug_label(other).0,
    };
    match state {
        TweetFieldsResultState::Found(found) => match &found.suppress_reason {
            None => label(Class::Allow, NONE),
            Some(suppress @ FilteredReason::SafetyResult(result)) => {
                (debug_label(&result.action).0, reason(suppress))
            }
            Some(suppress) => label(Class::Suppressed, reason(suppress)),
        },
        TweetFieldsResultState::Filtered(filtered) => {
            #[expect(
                clippy::wildcard_enum_match_arm,
                reason = "FilteredReason is a generated Thrift union; every variant but SafetyResult is a plain drop, including any the IDL adds"
            )]
            let drop_label = match &filtered.reason {
                FilteredReason::SafetyResult(result)
                    if result.reason.is_none()
                        && matches!(&result.action, Action::Drop(drop) if drop.reason.is_none()) =>
                {
                    label(Class::BareDrop, NONE)
                }
                filtered @ FilteredReason::SafetyResult(result) => {
                    (debug_label(&result.action).0, reason(filtered))
                }
                filtered => label(Class::Drop, reason(filtered)),
            };
            drop_label
        }
        TweetFieldsResultState::NotFound(not_found) => label(
            Class::NotFound,
            not_found
                .filtered_reason
                .as_ref()
                .map_or_else(|| NONE.to_string(), reason),
        ),
        TweetFieldsResultState::Failed(result) if result.over_capacity => {
            failed(Failure::OverCapacity)
        }
        TweetFieldsResultState::Failed(_) => failed(Failure::TweetypieFailed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_alike_matches_only_vf_bare_drop_to_an_unscreened_tweetypie_drop() {
        let bare_drop = label(Class::BareDrop, NONE);
        let deactivated = label(Class::Drop, "author_is_deactivated");
        let suspended = label(Class::Drop, "author_is_suspended");
        for (vf, tweetypie, alike) in [
            (&bare_drop, &suspended, false),
            (&bare_drop, &deactivated, true),
            (&deactivated, &bare_drop, false),
            (&suspended, &deactivated, false),
            (&suspended, &suspended, true),
        ] {
            assert_eq!(
                renders_alike(vf, tweetypie),
                alike,
                "vf {vf:?}, tweetypie {tweetypie:?}"
            );
        }
    }
}
