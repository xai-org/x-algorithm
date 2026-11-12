use super::proto::proto_action;
use super::thrift::thrift_result_state;
use crate::hydration::Lookup;
use crate::models::{Decided, Deletion, Evaluation, Verdict, VerdictAction};
use crate::params::LimitedActionsPolicies;
use crate::post_evaluator::PostEvaluation;
use crate::rules::{RuleId, SafetyLevel};
use std::sync::Arc;
use vf_pb::tweet_evaluation::Outcome as TweetEvaluationOutcome;
use vf_pb::NotificationDropReason;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::tweet_service::{TweetFieldsResultNotFound, TweetFieldsResultState};

pub(crate) fn verdict_action(verdict: &Verdict) -> VerdictAction {
    match verdict {
        Verdict::Withheld(decided) => decided.value.kind().verdict_action(),
        Verdict::Fosnr(decided) => decided.value.kind().verdict_action(),
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: None,
        } => VerdictAction::Allow,
        Verdict::Shown {
            label_interstitial: None,
            media: Some(_),
            engagement: None,
        } => VerdictAction::Interstitial,
        Verdict::Shown {
            label_interstitial: None,
            media: None,
            engagement: Some(_),
        } => VerdictAction::LimitedEngagement,
        Verdict::Shown { .. } => VerdictAction::TweetInterstitial,
    }
}

pub(crate) fn decided_rows(
    verdict: &Verdict,
) -> impl Iterator<Item = (&'static str, VerdictAction)> {
    verdict
        .decided()
        .map(|(kind, by)| (by, kind.verdict_action()))
}

pub(crate) fn decided_by(verdict: &Verdict) -> Option<&'static str> {
    decided_rows(verdict).next().map(|(rule, _)| rule)
}

pub(crate) fn filter_tweets_result(evaluated: PostEvaluation) -> vf_pb::TweetVisibilityResult {
    let (action, filtered_reason) = proto_action(evaluated.evaluation.verdict());
    vf_pb::TweetVisibilityResult {
        tweet_id: evaluated.tweet_id.0,
        action: Some(action),
        filtered_reason,
        safety_labels: evaluated.safety_labels.map(Arc::unwrap_or_clone),
    }
}

pub(crate) fn evaluate_tweets_outcome(
    evaluation: Option<&Evaluation>,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> TweetEvaluationOutcome {
    match evaluation
        .and_then(|evaluation| evaluate_tweets_state(evaluation, level, policies))
        .map(|state| xai_x_thrift::serialize_compact(&state))
    {
        Some(Ok(bytes)) => TweetEvaluationOutcome::ResultStateThriftCompact(bytes.into()),
        Some(Err(_)) | None => TweetEvaluationOutcome::Failed(vf_pb::Failed {}),
    }
}

pub(crate) fn evaluate_tweets_state(
    evaluation: &Evaluation,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Option<TweetFieldsResultState> {
    match evaluation {
        Evaluation::Complete { verdict, .. } => Some(thrift_result_state(verdict, level, policies)),
        Evaluation::NotFound(_) => Some(TweetFieldsResultState::NotFound(
            TweetFieldsResultNotFound::new(false, false, None),
        )),
        Evaluation::Deleted(deletion) => Some(TweetFieldsResultState::NotFound(
            TweetFieldsResultNotFound::new(true, *deletion == Deletion::BounceDeleted, None),
        )),
        Evaluation::Partial { .. } | Evaluation::Failed(_) => None,
    }
}

pub(crate) fn notification_result(
    evaluation: Option<&Evaluation>,
) -> (
    vf_pb::NotificationVisibilityResult,
    (&'static str, &'static str),
) {
    use vf_pb::notification_visibility_result::Outcome;
    let dropped = |reason: NotificationDropReason| {
        (
            Outcome::Dropped(vf_pb::Dropped {
                reason: Some(reason.into()),
            }),
            ("dropped", notification_reason_label(reason)),
        )
    };
    let (outcome, labels) = match evaluation {
        Some(
            Evaluation::Complete {
                verdict,
                fail_open_defaults,
            }
            | Evaluation::Partial {
                verdict,
                fail_open_defaults,
            },
        ) if fail_open_defaults.is_empty() => match verdict {
            Verdict::Withheld(Decided { id, .. }) => dropped(notification_drop_reason(*id)),
            Verdict::Fosnr(_) | Verdict::Shown { .. } => {
                (Outcome::Allowed(vf_pb::Allowed {}), ("allowed", "none"))
            }
        },
        Some(Evaluation::NotFound(Lookup::Author | Lookup::SharedAuthor)) => {
            dropped(NotificationDropReason::ActorNotFound)
        }
        Some(
            Evaluation::NotFound(Lookup::Tweet | Lookup::SharedTweet) | Evaluation::Deleted(_),
        ) => dropped(NotificationDropReason::PostNotFound),
        Some(Evaluation::Complete { .. } | Evaluation::Partial { .. } | Evaluation::Failed(_))
        | None => (Outcome::Failed(vf_pb::Failed {}), ("failed", "none")),
    };
    (
        vf_pb::NotificationVisibilityResult {
            outcome: Some(outcome),
        },
        labels,
    )
}

const NOTIFICATION_DROP_REASONS: [(RuleId, NotificationDropReason); 4] = [
    (
        RuleId::SuspendedAuthor,
        NotificationDropReason::ActorSuspended,
    ),
    (
        RuleId::ProtectedAuthor,
        NotificationDropReason::ActorProtected,
    ),
    (
        RuleId::ViewerBlocksAuthor,
        NotificationDropReason::RecipientBlocksActor,
    ),
    (
        RuleId::ViewerMutesAndDoesNotFollowAuthor,
        NotificationDropReason::RecipientMutesActor,
    ),
];

fn notification_drop_reason(rule: RuleId) -> NotificationDropReason {
    NOTIFICATION_DROP_REASONS
        .iter()
        .find(|&&(id, _)| id == rule)
        .map_or(NotificationDropReason::Unspecified, |&(_, reason)| reason)
}

pub(crate) fn notification_reason_label(reason: NotificationDropReason) -> &'static str {
    match reason {
        NotificationDropReason::Unspecified => "unspecified",
        NotificationDropReason::ActorNotFound => "actor_not_found",
        NotificationDropReason::ActorSuspended => "actor_suspended",
        NotificationDropReason::ActorProtected => "actor_protected",
        NotificationDropReason::RecipientBlocksActor => "recipient_blocks_actor",
        NotificationDropReason::RecipientMutesActor => "recipient_mutes_actor",
        NotificationDropReason::PostNotFound => "post_not_found",
    }
}

#[cfg(test)]
mod tests;
