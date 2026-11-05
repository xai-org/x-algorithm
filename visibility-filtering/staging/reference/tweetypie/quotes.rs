use super::{
    get_tweet_fields, BatchOutcome, Compared, Lens, TpResult, TweetypieReference, TWEETYPIE_TIMEOUT,
};
use crate::models::viewer::Client;
use crate::models::{EvaluationKey, Placement, RequestViewer, TweetId};
use crate::post_evaluator::PostEvaluation;
use crate::quote::QuotePass;
use crate::result_label::{label, Class, NONE};
use crate::rules::SafetyLevel;
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
    pub(in crate::staging::reference) fn spawn_entry_comparison(
        self: &Arc<Self>,
        viewer: &RequestViewer,
        safety_level: SafetyLevel,
        outcomes: &[PostEvaluation],
        quote_passes: &[QuotePass],
    ) {
        let Some(viewer_id) = viewer.id() else {
            return;
        };
        if !self.sample.selects(viewer_id, outcomes) {
            return;
        }
        let Some(entries) = entry_comparisons(safety_level, outcomes, quote_passes) else {
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
                .compare_entries(viewer_id, country_code, client, entries)
                .await
        });
    }

    async fn compare_entries(
        &self,
        viewer_id: u64,
        country_code: Option<String>,
        client: TwitterContextViewer,
        entries: EntryComparisons,
    ) -> BatchOutcome {
        let Ok(tp) = timeout(
            TWEETYPIE_TIMEOUT,
            get_tweet_fields(&self.tweetypie, &client, &entries.fetched),
        )
        .await
        else {
            return BatchOutcome::TweetypieTimeout;
        };
        let by_id: HashMap<TweetId, &TpResult> = entries.fetched.iter().copied().zip(&tp).collect();
        let answers: Vec<(&Served, TpResult)> = entries
            .served
            .iter()
            .filter_map(|served| Some((served, served.lens.answer(by_id.get(&served.fetched)?))))
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
    fetched: TweetId,
    outcome: PostEvaluation,
    pass: Option<QuotePass>,
}

struct EntryComparisons {
    fetched: Vec<TweetId>,
    served: Vec<Served>,
}

fn entry_comparisons(
    safety_level: SafetyLevel,
    outcomes: &[PostEvaluation],
    quote_passes: &[QuotePass],
) -> Option<EntryComparisons> {
    match safety_level {
        SafetyLevel::TimelineHomeHydration => {}
        SafetyLevel::FilterAll
        | SafetyLevel::TimelineHome
        | SafetyLevel::TimelineHomeRecommendations
        | SafetyLevel::TimelineHomePromotedHydration
        | SafetyLevel::ImmersiveExpandedRecommendations
        | SafetyLevel::QuotedTweetRules
        | SafetyLevel::NotificationsFollow
        | SafetyLevel::NotificationsWriterV2 => return None,
    }
    let outers: HashSet<TweetId> = outcomes
        .iter()
        .filter_map(|outcome| match outcome.placement {
            Placement::InnerQuoted { outer } => Some(outer),
            Placement::Standalone | Placement::RetweetSource => None,
        })
        .collect();
    let passes: HashMap<EvaluationKey, &QuotePass> =
        quote_passes.iter().map(|pass| (pass.key, pass)).collect();
    let mut seen = HashSet::new();
    let mut fetched = Vec::new();
    let mut served = Vec::new();
    for outcome in outcomes {
        let (lens, id) = match outcome.placement {
            Placement::InnerQuoted { outer } => (Lens::InnerQuoted, outer),
            Placement::Standalone if outers.contains(&outcome.tweet_id) => {
                (Lens::Outer, outcome.tweet_id)
            }
            Placement::Standalone => (Lens::Standalone, outcome.tweet_id),
            Placement::RetweetSource => continue,
        };
        if !seen.insert(outcome.key()) {
            continue;
        }
        if !fetched.contains(&id) {
            fetched.push(id);
        }
        let pass = passes.get(&outcome.key()).map(|&pass| pass.clone());
        if let Some(pass) = &pass {
            served.push(Served {
                lens: Lens::QuotePass,
                fetched: id,
                outcome: outcome.clone(),
                pass: Some(pass.clone()),
            });
        }
        served.push(Served {
            lens,
            fetched: id,
            outcome: outcome.clone(),
            pass,
        });
    }
    (!served.is_empty()).then_some(EntryComparisons { fetched, served })
}

#[cfg(test)]
mod tests {
    use super::super::tests::{compared, counted, outcome, pair};
    use super::super::{tp_failed, tp_result, SAFETY_LEVEL};
    use super::*;
    use crate::hydration::{Hydrators, Lookup};
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

    fn entry(tweet_id: u64, placement: Placement) -> PostEvaluation {
        PostEvaluation {
            tweet_id: TweetId(tweet_id),
            placement,
            ..outcome(allow())
        }
    }

    #[test]
    fn only_a_level_82_request_compares_each_entry_once_in_one_fetch() {
        let quoting = |outer| Placement::InnerQuoted {
            outer: TweetId(outer),
        };
        let request = [
            entry(10, Placement::Standalone),
            entry(1, quoting(10)),
            entry(3, Placement::Standalone),
            entry(1, quoting(10)),
            entry(4, Placement::RetweetSource),
            entry(2, quoting(20)),
        ];
        let selected = |level, outcomes: &[PostEvaluation]| {
            entry_comparisons(level, outcomes, &[]).map(|entries| {
                let served: Vec<_> = entries
                    .served
                    .iter()
                    .map(|served| (served.lens, served.fetched.0, served.outcome.tweet_id.0))
                    .collect();
                (entries.fetched, served)
            })
        };
        assert_eq!(
            selected(SAFETY_LEVEL, &request),
            Some((
                vec![TweetId(10), TweetId(3), TweetId(20)],
                vec![
                    (Lens::Outer, 10, 10),
                    (Lens::InnerQuoted, 10, 1),
                    (Lens::Standalone, 3, 3),
                    (Lens::InnerQuoted, 20, 2),
                ]
            ))
        );
        for &level in SafetyLevel::VARIANTS {
            if level != SAFETY_LEVEL {
                assert!(selected(level, &request).is_none(), "{level:?}");
            }
        }
        assert_eq!(
            selected(SAFETY_LEVEL, &request[2..3]),
            Some((vec![TweetId(3)], vec![(Lens::Standalone, 3, 3)]))
        );
        let passes = [QuotePass {
            key: request[1].key(),
            outcome: PassOutcome::Passed,
            author: None,
        }];
        let lenses: Vec<_> = entry_comparisons(SAFETY_LEVEL, &request, &passes)
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
                (Lens::Standalone, 3, false),
                (Lens::InnerQuoted, 2, false),
            ]
        );
    }

    #[test]
    fn a_withheld_partial_or_failed_served_entry_is_compared() {
        let suspended = Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(VfFilteredReason::AuthorIsSuspended)),
            id: RuleId::SuspendedAuthor,
            by: "suspended_author/drop",
        });
        let request = [
            PostEvaluation {
                evaluation: Evaluation::complete(suspended),
                ..entry(1, Placement::Standalone)
            },
            PostEvaluation {
                evaluation: Evaluation::Partial {
                    verdict: allow(),
                    fail_open_defaults: Hydrators::empty(),
                },
                ..entry(2, Placement::Standalone)
            },
            PostEvaluation {
                evaluation: Evaluation::Failed(Lookup::Author),
                ..entry(3, Placement::Standalone)
            },
        ];
        let entries = entry_comparisons(SafetyLevel::TimelineHomeHydration, &request, &[]).unwrap();
        assert_eq!(entries.fetched, [TweetId(1), TweetId(2), TweetId(3)]);
        let tp = TpResult {
            label: pair("allow", NONE),
            result: None,
            strato_error: None,
            quoted: None,
        };
        let vf: Vec<_> = entries
            .served
            .iter()
            .map(|served| {
                Compared {
                    lens: served.lens,
                    ..compared(&tp, Some(&served.outcome))
                }
                .bucket()
                .vf
            })
            .collect();
        assert_eq!(
            vf,
            [
                pair("drop", "author_is_suspended"),
                pair("failed", NONE),
                pair("failed", NONE),
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
