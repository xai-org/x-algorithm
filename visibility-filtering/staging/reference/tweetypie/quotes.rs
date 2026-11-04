use super::{
    BatchOutcome, Compared, Lens, TWEETYPIE_TIMEOUT, TpResult, TweetypieReference, get_tweet_fields,
};
use crate::models::viewer::Client;
use crate::models::{EvaluationKey, Placement, RequestViewer, TweetId};
use crate::post_evaluator::PostEvaluation;
use crate::quote::QuotePass;
use crate::result_label::{Class, NONE, label};
use crate::rules::{QuoteEntries, SafetyLevel};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::time::timeout;
use xai_twittercontext_proto::TwitterContextViewer;

impl Lens {
    fn answer(self, outer: &TpResult) -> TpResult {
        let failed_class: &str = Class::Failed.into();
        match (self, &outer.quoted) {
            (Self::Standalone | Self::Outer, _) => outer.clone(),
            (Self::InnerQuoted | Self::QuotePass, Some(quoted)) => (**quoted).clone(),
            (Self::InnerQuoted | Self::QuotePass, None) => TpResult {
                label: if outer.label.0 == failed_class {
                    outer.label.clone()
                } else {
                    label(Class::Missing, NONE)
                },
                result: None,
                strato_error: outer.strato_error.clone(),
                quoted: None,
            },
        }
    }
}

impl TweetypieReference {
    pub(in crate::staging::reference) fn spawn_quote_comparison(
        self: &Arc<Self>,
        viewer: &RequestViewer,
        safety_level: SafetyLevel,
        outcomes: &[PostEvaluation],
        quote_passes: &[QuotePass],
    ) {
        let Some(viewer_id) = viewer.id() else {
            return;
        };
        let Some(quotes) = quote_comparisons(safety_level, outcomes, quote_passes) else {
            return;
        };
        let client = TwitterContextViewer {
            user_id: viewer_id.cast_signed(),
            ..viewer
                .context()
                .cloned()
                .unwrap_or_else(|| Client::Web.context(None, viewer.country_code()))
        };
        let country_code = viewer.country_code().map(str::to_owned);
        self.detach(move |reference| async move {
            reference
                .compare_quotes(viewer_id, country_code, client, quotes)
                .await
        });
    }

    async fn compare_quotes(
        &self,
        viewer_id: u64,
        country_code: Option<String>,
        client: TwitterContextViewer,
        quotes: QuoteComparisons,
    ) -> BatchOutcome {
        let Ok(tp) = timeout(
            TWEETYPIE_TIMEOUT,
            get_tweet_fields(&self.tweetypie, &client, &quotes.outers),
        )
        .await
        else {
            return BatchOutcome::TweetypieTimeout;
        };
        let by_outer: HashMap<TweetId, &TpResult> =
            quotes.outers.iter().copied().zip(&tp).collect();
        let answers: Vec<(&Served, TpResult)> = quotes
            .served
            .iter()
            .filter_map(|served| Some((served, served.lens.answer(by_outer.get(&served.outer)?))))
            .collect();
        self.record(answers.iter().map(|(served, tp)| Compared {
            lens: served.lens,
            tweet_id: served.outcome.tweet_id,
            viewer_id,
            country_code: country_code.as_deref(),
            tp,
            vf: Some(&served.outcome),
            pass: served.pass.as_ref(),
        }));
        BatchOutcome::Compared
    }
}

struct Served {
    lens: Lens,
    outer: TweetId,
    outcome: PostEvaluation,
    pass: Option<QuotePass>,
}

struct QuoteComparisons {
    outers: Vec<TweetId>,
    served: Vec<Served>,
}

fn quote_comparisons(
    safety_level: SafetyLevel,
    outcomes: &[PostEvaluation],
    quote_passes: &[QuotePass],
) -> Option<QuoteComparisons> {
    if !matches!(
        safety_level.spec().quote_entries(),
        QuoteEntries::Judged { .. }
    ) {
        return None;
    }
    let mut outers = Vec::new();
    for outcome in outcomes {
        if let Placement::InnerQuoted { outer } = outcome.placement
            && !outers.contains(&outer)
        {
            outers.push(outer);
        }
    }
    if outers.is_empty() {
        return None;
    }
    let passes: HashMap<EvaluationKey, &QuotePass> =
        quote_passes.iter().map(|pass| (pass.key, pass)).collect();
    let mut seen = HashSet::new();
    let mut served = Vec::new();
    for outcome in outcomes {
        let (lens, outer) = match outcome.placement {
            Placement::InnerQuoted { outer } => (Lens::InnerQuoted, outer),
            Placement::Standalone if outers.contains(&outcome.tweet_id) => {
                (Lens::Outer, outcome.tweet_id)
            }
            Placement::Standalone | Placement::RetweetSource => continue,
        };
        if !seen.insert(outcome.key()) {
            continue;
        }
        let pass = passes.get(&outcome.key()).map(|&pass| pass.clone());
        if let Some(pass) = &pass {
            served.push(Served {
                lens: Lens::QuotePass,
                outer,
                outcome: outcome.clone(),
                pass: Some(pass.clone()),
            });
        }
        served.push(Served {
            lens,
            outer,
            outcome: outcome.clone(),
            pass,
        });
    }
    Some(QuoteComparisons { outers, served })
}

#[cfg(test)]
mod tests {
    use super::super::tests::{compared, counted, outcome, pair};
    use super::super::{tp_failed, tp_result};
    use super::*;
    use crate::models::{Decided, DropReason, Evaluation, Verdict, Withholding};
    use crate::quote::{PassOutcome, Rerendered};
    use crate::result_label::Failure;
    use crate::rules::fixtures::allow;
    use crate::rules::{RerenderReason, RuleId};
    use strum::VariantArray;
    use xai_strato::strato_thrift::StratoResult;
    use xai_visibility_filtering::models::FilteredReason as VfFilteredReason;
    use xai_x_thrift::get_tweet_fields::{
        GetTweetFieldsResult, TweetFieldsResultFound, TweetFieldsResultState,
    };
    use xai_x_thrift::safety_result::FilteredReason;
    use xai_x_thrift::tweet_service::TweetFieldsResultFiltered;

    #[test]
    fn a_quote_entry_reads_its_outers_quoted_tweet_result_else_missing_or_the_outers_failure() {
        let found = TweetFieldsResultState::Found(TweetFieldsResultFound::new(None));
        let suspended = TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(
            FilteredReason::AuthorIsSuspended(true),
        ));
        let outer = |state: &TweetFieldsResultState, quoted| {
            tp_result(StratoResult::Ok {
                value: Some(GetTweetFieldsResult::new(5, state.clone(), quoted)),
                ttl: -1,
            })
        };
        let cases = [
            (
                outer(&found, Some(suspended.clone())),
                Lens::InnerQuoted,
                ("drop", "author_is_suspended"),
            ),
            (
                outer(&found, Some(suspended.clone())),
                Lens::Outer,
                ("allow", "none"),
            ),
            (outer(&found, None), Lens::InnerQuoted, ("missing", "none")),
            (
                outer(&suspended, None),
                Lens::InnerQuoted,
                ("missing", "none"),
            ),
            (
                tp_failed(Failure::RpcError),
                Lens::InnerQuoted,
                ("failed", "rpc_error"),
            ),
        ];
        for (outer, lens, (class, reason)) in cases {
            assert_eq!(
                lens.answer(&outer).label,
                pair(class, reason),
                "{lens:?} {:?}",
                outer.label
            );
        }
    }

    #[test]
    fn a_quote_entry_is_labelled_with_the_quoted_tweet_rules_rule_that_drops_its_author() {
        let tp = TpResult {
            label: pair("allow", NONE),
            result: None,
            strato_error: None,
            quoted: None,
        };
        let vf = PostEvaluation {
            placement: Placement::InnerQuoted { outer: TweetId(7) },
            ..outcome(allow())
        };
        let muted = Evaluation::complete(Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(VfFilteredReason::ViewerMutesAuthor)),
            id: RuleId::QuotedViewerMutesAuthorNotFollowing,
            by: "quoted_viewer_mutes_author_not_following/drop",
        }));
        let allowed = Evaluation::complete(allow());
        let pass = |author| QuotePass {
            key: vf.key(),
            outcome: PassOutcome::Passed,
            author,
        };
        for (pass, rule) in [
            (
                Some(pass(Some(muted.clone()))),
                "quoted_viewer_mutes_author_not_following/drop",
            ),
            (Some(pass(Some(allowed.clone()))), NONE),
            (Some(pass(None)), NONE),
            (None, NONE),
        ] {
            let compared = Compared {
                lens: Lens::InnerQuoted,
                pass: pass.as_ref(),
                ..compared(&tp, Some(&vf))
            };
            let bucket = compared.bucket();
            assert_eq!(counted(&bucket).label("quoted_tweet_rule"), Some(rule));
            assert_eq!(
                compared.to_json(&bucket, "sha", None)["quoted_tweet_rule"],
                rule
            );
        }
    }

    #[test]
    fn only_a_level_82_request_with_a_quote_entry_compares_each_entry_once_against_its_outer() {
        let entry = |tweet_id, placement| PostEvaluation {
            tweet_id: TweetId(tweet_id),
            placement,
            ..outcome(allow())
        };
        let quoting = |outer| Placement::InnerQuoted {
            outer: TweetId(outer),
        };
        let request = [
            entry(10, Placement::Standalone),
            entry(1, quoting(10)),
            entry(3, Placement::Standalone),
            entry(1, quoting(10)),
            entry(2, quoting(20)),
        ];
        let selected = |level, outcomes: &[PostEvaluation]| {
            quote_comparisons(level, outcomes, &[]).map(|quotes| {
                let served: Vec<_> = quotes
                    .served
                    .iter()
                    .map(|served| (served.lens, served.outer.0, served.outcome.tweet_id.0))
                    .collect();
                (quotes.outers, served)
            })
        };
        assert_eq!(
            selected(SafetyLevel::TimelineHomeHydration, &request),
            Some((
                vec![TweetId(10), TweetId(20)],
                vec![
                    (Lens::Outer, 10, 10),
                    (Lens::InnerQuoted, 10, 1),
                    (Lens::InnerQuoted, 20, 2),
                ]
            ))
        );
        for &level in SafetyLevel::VARIANTS {
            if !matches!(level.spec().quote_entries(), QuoteEntries::Judged { .. }) {
                assert!(selected(level, &request).is_none(), "{level:?}");
            }
        }
        assert!(selected(SafetyLevel::TimelineHomeHydration, &request[..1]).is_none());
        let passes = [QuotePass {
            key: request[1].key(),
            outcome: PassOutcome::Passed,
            author: None,
        }];
        let lenses: Vec<_> =
            quote_comparisons(SafetyLevel::TimelineHomeHydration, &request, &passes)
                .unwrap()
                .served
                .iter()
                .map(|served| {
                    (
                        served.lens,
                        served.outcome.tweet_id.0,
                        served.pass.is_some(),
                    )
                })
                .collect();
        assert_eq!(
            lenses,
            [
                (Lens::Outer, 10, false),
                (Lens::QuotePass, 1, true),
                (Lens::InnerQuoted, 1, true),
                (Lens::InnerQuoted, 2, false),
            ]
        );
    }

    #[test]
    fn the_quote_pass_lens_compares_the_pass_result_and_names_its_outcome() {
        let tp = TpResult {
            label: pair("drop", "author_is_suspended"),
            result: None,
            strato_error: None,
            quoted: None,
        };
        let inner = PostEvaluation {
            placement: Placement::InnerQuoted { outer: TweetId(7) },
            ..outcome(allow())
        };
        let suspended = Evaluation::complete(Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(VfFilteredReason::AuthorIsSuspended)),
            id: RuleId::SuspendedAuthor,
            by: "suspended_author/drop",
        }));
        let dropped = PassOutcome::Dropped(Rerendered {
            by: RuleId::SuspendedAuthor,
            reason: RerenderReason::AuthorSuspended,
        });
        for (outcome, lens, vf, quote_pass) in [
            (
                dropped.clone(),
                Lens::QuotePass,
                pair("drop", "author_is_suspended"),
                "dropped",
            ),
            (dropped, Lens::InnerQuoted, pair("allow", NONE), NONE),
            (
                PassOutcome::OuterWithheld,
                Lens::QuotePass,
                pair("missing", NONE),
                "outer_withheld",
            ),
            (
                PassOutcome::FailedOpen,
                Lens::QuotePass,
                pair("allow", NONE),
                "failed_open",
            ),
        ] {
            let pass = QuotePass {
                key: inner.key(),
                outcome,
                author: Some(suspended.clone()),
            };
            let compared = Compared {
                lens,
                pass: Some(&pass),
                ..compared(&tp, Some(&inner))
            };
            let bucket = compared.bucket();
            assert_eq!(
                (&bucket.vf, bucket.quote_pass, bucket.quoted_tweet_rule),
                (&vf, quote_pass, "suspended_author/drop"),
                "{lens:?} {:?}",
                pass.outcome
            );
            assert_eq!(counted(&bucket).label("quote_pass"), Some(quote_pass));
            assert_eq!(
                compared.to_json(&bucket, "sha", None)["quote_pass"],
                quote_pass
            );
        }
    }
}
