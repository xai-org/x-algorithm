pub mod metrics;
pub mod nsfw_age_gating;
pub mod nsfw_interstitial;
pub mod nullcast_rule;
pub mod registry;
pub mod socialgraph_rules;
pub mod tes_rules;
pub mod tweet_flag_rules;
pub mod tweet_label_drops;
pub mod user_label_drops;
pub mod user_rules;
pub mod high_risk_tag_drops;

use crate::models::{HydratedTweetCandidate, SafetyLabelType, VfAction, ViewerFeatures};
use xai_visibility_filtering::models::FilteredReason;

pub use registry::{Policies, SafetyLevel};

pub struct RuleContext<'a> {
    safety_level: SafetyLevel,
    viewer: &'a ViewerFeatures,
    candidate: &'a HydratedTweetCandidate,
}

impl<'a> RuleContext<'a> {
    fn new(
        safety_level: SafetyLevel,
        viewer: &'a ViewerFeatures,
        candidate: &'a HydratedTweetCandidate,
    ) -> Self {
        Self {
            safety_level,
            viewer,
            candidate,
        }
    }

    pub fn safety_level(&self) -> SafetyLevel {
        self.safety_level
    }

    pub fn has_tweet_safety_label(&self, label: SafetyLabelType) -> bool {
        self.candidate.has_safety_label(label)
    }

    pub fn is_author_viewer(&self) -> bool {
        self.candidate.is_author_viewer(self.viewer.viewer)
    }

    pub fn viewer_follows_author(&self) -> bool {
        self.candidate.viewer_follows_author()
    }

    pub fn viewer_allows_sensitive_media(&self) -> bool {
        self.viewer.allows_sensitive_media
    }

    pub fn viewer(&self) -> &ViewerFeatures {
        self.viewer
    }

    pub fn candidate(&self) -> &HydratedTweetCandidate {
        self.candidate
    }
}

pub trait Rule: Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction;
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub action: VfAction,
    pub decided_by: Option<&'static str>,
}

impl Verdict {
    pub fn unresolved_author() -> Self {
        Self {
            action: VfAction::Drop(FilteredReason::UnspecifiedReason),
            decided_by: Some("unresolved_author_id"),
        }
    }
}

fn evaluate_rules(rules: &[Box<dyn Rule>], context: &RuleContext<'_>) -> Verdict {
    let mut worst = VfAction::Allow;
    let mut decided_by = None;
    for rule in rules {
        match rule.evaluate(context) {
            VfAction::Drop(reason) => {
                return Verdict {
                    action: VfAction::Drop(reason),
                    decided_by: Some(rule.name()),
                };
            }
            VfAction::Interstitial(reason) => {
                if matches!(worst, VfAction::Allow) {
                    worst = VfAction::Interstitial(reason);
                    decided_by = Some(rule.name());
                }
            }
            VfAction::Allow => {}
        }
    }
    Verdict {
        action: worst,
        decided_by,
    }
}

#[cfg(test)]
pub(crate) fn test_context<'a>(
    viewer: &'a ViewerFeatures,
    candidate: &'a HydratedTweetCandidate,
) -> RuleContext<'a> {
    RuleContext::new(SafetyLevel::TimelineHome, viewer, candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct FakeRule {
        name: &'static str,
        action: VfAction,
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Rule for FakeRule {
        fn name(&self) -> &'static str {
            self.name
        }

        fn evaluate(&self, _context: &RuleContext<'_>) -> VfAction {
            self.calls.lock().unwrap().push(self.name);
            self.action.clone()
        }
    }

    type CallLog = Arc<Mutex<Vec<&'static str>>>;

    fn recording_rules(specs: &[(&'static str, VfAction)]) -> (Vec<Box<dyn Rule>>, CallLog) {
        let calls = CallLog::default();
        let rules = specs
            .iter()
            .map(|(name, action)| {
                Box::new(FakeRule {
                    name,
                    action: action.clone(),
                    calls: calls.clone(),
                }) as Box<dyn Rule>
            })
            .collect();
        (rules, calls)
    }

    fn context_inputs() -> (ViewerFeatures, HydratedTweetCandidate) {
        (ViewerFeatures::default(), HydratedTweetCandidate::default())
    }

    #[test]
    fn drop_short_circuits_later_rules() {
        let (rules, calls) = recording_rules(&[
            ("allow", VfAction::Allow),
            ("drop", VfAction::Drop(FilteredReason::AuthorIsSuspended)),
            ("after_drop", VfAction::Allow),
        ]);
        let (viewer, candidate) = context_inputs();

        let verdict = evaluate_rules(&rules, &test_context(&viewer, &candidate));

        assert!(matches!(
            verdict.action,
            VfAction::Drop(FilteredReason::AuthorIsSuspended)
        ));
        assert_eq!(verdict.decided_by, Some("drop"));
        assert_eq!(*calls.lock().unwrap(), vec!["allow", "drop"]);
    }

    #[test]
    fn first_interstitial_decided_by_sticks_without_short_circuit() {
        let (rules, calls) = recording_rules(&[
            (
                "first_interstitial",
                VfAction::Interstitial(FilteredReason::ContainNsfwMedia),
            ),
            (
                "second_interstitial",
                VfAction::Interstitial(FilteredReason::UnspecifiedReason),
            ),
            ("after_interstitials", VfAction::Allow),
        ]);
        let (viewer, candidate) = context_inputs();

        let verdict = evaluate_rules(&rules, &test_context(&viewer, &candidate));

        assert!(matches!(
            verdict.action,
            VfAction::Interstitial(FilteredReason::ContainNsfwMedia)
        ));
        assert_eq!(verdict.decided_by, Some("first_interstitial"));
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                "first_interstitial",
                "second_interstitial",
                "after_interstitials"
            ]
        );
    }

    #[test]
    fn drop_after_interstitial_wins() {
        let (rules, _) = recording_rules(&[
            (
                "interstitial",
                VfAction::Interstitial(FilteredReason::ContainNsfwMedia),
            ),
            ("drop", VfAction::Drop(FilteredReason::AuthorIsSuspended)),
        ]);
        let (viewer, candidate) = context_inputs();

        let verdict = evaluate_rules(&rules, &test_context(&viewer, &candidate));

        assert!(matches!(
            verdict.action,
            VfAction::Drop(FilteredReason::AuthorIsSuspended)
        ));
        assert_eq!(verdict.decided_by, Some("drop"));
    }

    #[test]
    fn all_allows_is_allow_with_no_decider() {
        let (rules, _) = recording_rules(&[("a", VfAction::Allow), ("b", VfAction::Allow)]);
        let (viewer, candidate) = context_inputs();

        let verdict = evaluate_rules(&rules, &test_context(&viewer, &candidate));

        assert!(matches!(verdict.action, VfAction::Allow));
        assert_eq!(verdict.decided_by, None);
    }
}
