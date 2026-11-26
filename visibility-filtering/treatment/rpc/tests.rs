use super::*;
use crate::result_label::{label, state_label, Class, NONE};
use crate::rules::fixtures::{allow, dropped};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::safety_result::FilteredReason as ThriftFilteredReason;
use xai_x_thrift::tweet_service::{TweetFieldsResultFiltered, TweetFieldsResultFound};

#[test]
fn decided_by_names_the_deciding_rule_or_why_the_rules_never_ran() {
    use crate::hydration::{Hydrator, Hydrators};
    let suspended = dropped(FilteredReason::AuthorIsSuspended, "rule");
    for (evaluation, rule) in [
        (Evaluation::complete(allow()), None),
        (Evaluation::complete(suspended.clone()), Some("rule")),
        (
            Evaluation::Partial {
                verdict: suspended,
                fail_open_defaults: Hydrators::of(Hydrator::Follows),
            },
            Some("rule"),
        ),
        (Evaluation::NotFound(Lookup::Author), Some("not_found")),
        (Evaluation::Failed(Lookup::Tweet), Some("lookup_failed")),
    ] {
        assert_eq!(decided_by(evaluation.verdict()), rule, "{evaluation:?}");
    }
}

#[test]
fn dashboard_generator_pins_the_verdict_mix_labels() {
    use strum::VariantArray;
    let emitted: std::collections::BTreeSet<String> = VerdictAction::VARIANTS
        .iter()
        .map(|&action| <&str>::from(action).to_owned())
        .collect();
    let dashboard = crate::pinned::dashboard_tuple("FT_VERDICT_ACTIONS");
    assert_eq!(dashboard, emitted);
}

#[test]
fn each_evaluation_sends_and_labels_as_vf_counts_it() {
    use crate::hydration::{Hydrator, Hydrators};
    let shown = TweetFieldsResultState::Found(TweetFieldsResultFound::new(None));
    let suspended = TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(
        ThriftFilteredReason::AuthorIsSuspended(true),
    ));
    let not_found =
        TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(false, false, None));
    let cases = [
        (
            Evaluation::complete(allow()),
            Some(&shown),
            ("allow", "none"),
        ),
        (
            Evaluation::complete(dropped(FilteredReason::AuthorIsSuspended, "rule")),
            Some(&suspended),
            ("drop", "author_is_suspended"),
        ),
        (
            Evaluation::Partial {
                verdict: dropped(FilteredReason::AuthorIsSuspended, "rule"),
                fail_open_defaults: Hydrators::of(Hydrator::Follows),
            },
            None,
            ("failed", "none"),
        ),
        (
            Evaluation::NotFound(Lookup::Author),
            Some(&not_found),
            ("not_found", "none"),
        ),
        (Evaluation::Failed(Lookup::Tweet), None, ("failed", "none")),
    ];
    let policies = LimitedActionsPolicies::default();
    let level = SafetyLevel::TimelineHomeHydration;
    for (evaluation, sent, (class, reason)) in &cases {
        let state = evaluate_tweets_state(evaluation, level, &policies);
        assert_eq!(state.as_ref(), *sent, "{evaluation:?}");
        assert_eq!(
            state.map_or_else(|| label(Class::Failed, NONE), |state| state_label(&state)),
            ((*class).to_owned(), (*reason).to_owned()),
            "{evaluation:?}"
        );
    }
}
