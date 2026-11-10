use crate::hydration::{request_context, Cause, Unresolved};
use crate::metrics::families::QUOTE_PASS;
use crate::metrics::{self as ft_metrics, RequestMetrics};
use crate::models::{
    Decided, Evaluation, EvaluationKey, PlacedCandidate, Placement, RetweetMerge, Verdict,
};
use crate::post_evaluator::{AuthorJudgement, PostBatch, PostEvaluation, PostEvaluator};
use crate::rules::{NoTrace, QuoteEntries, RerenderReason, RuleId, SafetyLevel};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::Instant;

const SHED: &str = "shed";
const MAX_SHADOW_PASSES_IN_FLIGHT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuotePassMode {
    Off,
    Shadow,
}

#[derive(Clone, Debug, PartialEq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PassOutcome {
    OuterWithheld,
    OuterAbsent,
    InnerUnresolved,
    FailedOpen,
    NotRerendered,
    Dropped(Rerendered),
    Passed,
}

impl PassOutcome {
    pub(crate) fn rendering(&self) -> QuoteRendering<'_> {
        match self {
            Self::OuterWithheld => QuoteRendering::Omitted,
            Self::Dropped(rerendered) => QuoteRendering::Rerendered(rerendered),
            Self::OuterAbsent
            | Self::InnerUnresolved
            | Self::FailedOpen
            | Self::NotRerendered
            | Self::Passed => QuoteRendering::Served,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Rerendered {
    pub(crate) by: RuleId,
    pub(crate) reason: RerenderReason,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum QuoteRendering<'a> {
    Served,
    Rerendered(&'a Rerendered),
    Omitted,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct QuotePass {
    pub(crate) key: EvaluationKey,
    pub(crate) outcome: PassOutcome,
    pub(crate) author: Option<Evaluation>,
}

fn quoted_authors(request: &PostBatch) -> Option<PostBatch> {
    let quotes: Vec<PlacedCandidate> = request
        .candidates
        .iter()
        .filter(|placed| matches!(placed.placement, Placement::InnerQuoted { .. }))
        .copied()
        .collect();
    (!quotes.is_empty()).then(|| PostBatch {
        facts: request.facts.clone(),
        safety_level: SafetyLevel::QuotedTweetRules,
        candidates: quotes,
        retweet_merge: RetweetMerge::Unmerged,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Passes {
    Skipped,
    Shed,
    Ran(Vec<QuotePass>),
}

impl Passes {
    pub(crate) fn ran(&self) -> &[QuotePass] {
        match self {
            Self::Ran(passes) => passes,
            Self::Skipped | Self::Shed => &[],
        }
    }
}

pub(crate) struct QuotePassRunner {
    mode: QuotePassMode,
    permits: Arc<Semaphore>,
}

pub(crate) struct PendingPass {
    level: SafetyLevel,
    authors: Option<PostBatch>,
    metrics: RequestMetrics,
}

impl QuotePassRunner {
    pub(crate) fn new(mode: QuotePassMode) -> Self {
        Self {
            mode,
            permits: Arc::new(Semaphore::new(MAX_SHADOW_PASSES_IN_FLIGHT)),
        }
    }

    pub(crate) async fn serve(
        &self,
        post_evaluator: &PostEvaluator,
        request: PostBatch,
        metrics: &RequestMetrics,
    ) -> (Vec<PostEvaluation>, PendingPass) {
        let pending = PendingPass {
            level: request.safety_level,
            authors: match (self.mode, request.safety_level.spec().quote_entries()) {
                (QuotePassMode::Shadow, QuoteEntries::JudgedWithQuotePass { .. }) => {
                    quoted_authors(&request)
                }
                (QuotePassMode::Off, _)
                | (QuotePassMode::Shadow, QuoteEntries::NotJudged | QuoteEntries::Judged) => None,
            },
            metrics: metrics.at_level(SafetyLevel::QuotedTweetRules),
        };
        let (_, outcomes) = post_evaluator.run(request, metrics, &mut NoTrace).await;
        (outcomes, pending)
    }

    pub(crate) fn finish(
        &self,
        post_evaluator: &Arc<PostEvaluator>,
        pending: PendingPass,
        outcomes: Vec<PostEvaluation>,
        then: impl FnOnce(&[PostEvaluation], &Passes) + Send + 'static,
    ) {
        let Some(authors) = &pending.authors else {
            return then(&outcomes, &Passes::Skipped);
        };
        let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
            pending.metrics.count(
                &QUOTE_PASS,
                [SHED],
                u64::try_from(authors.candidates.len()).unwrap_or(u64::MAX),
            );
            return then(&outcomes, &Passes::Shed);
        };
        let post_evaluator = Arc::clone(post_evaluator);
        tokio::spawn(async move {
            let _permit = permit;
            let passes = request_context(Instant::now(), None)
                .scope(pending.run(&post_evaluator, &outcomes))
                .await;
            then(&outcomes, &passes);
        });
    }
}

impl PendingPass {
    pub(crate) async fn run(
        self,
        post_evaluator: &PostEvaluator,
        outcomes: &[PostEvaluation],
    ) -> Passes {
        match self.authors {
            None => Passes::Skipped,
            Some(authors) => Passes::Ran(
                quote_pass(post_evaluator, self.level, authors, outcomes, &self.metrics).await,
            ),
        }
    }
}

async fn quote_pass(
    post_evaluator: &PostEvaluator,
    level: SafetyLevel,
    authors: PostBatch,
    outcomes: &[PostEvaluation],
    metrics: &RequestMetrics,
) -> Vec<QuotePass> {
    let authors = post_evaluator.evaluate_authors(authors, metrics).await;
    let passes = passes(level, outcomes, &authors);
    record(metrics, &passes);
    passes
}

fn passes(
    level: SafetyLevel,
    outcomes: &[PostEvaluation],
    authors: &HashMap<EvaluationKey, AuthorJudgement>,
) -> Vec<QuotePass> {
    let outers: HashMap<_, _> = outcomes
        .iter()
        .filter(|outcome| outcome.placement == Placement::Standalone)
        .map(|outcome| (outcome.tweet_id, &outcome.evaluation))
        .collect();
    outcomes
        .iter()
        .filter_map(|outcome| {
            let Placement::InnerQuoted { outer } = outcome.placement else {
                return None;
            };
            let author = authors.get(&outcome.key());
            Some(QuotePass {
                key: outcome.key(),
                outcome: pass(
                    level,
                    outers.get(&outer).copied(),
                    &outcome.evaluation,
                    author,
                ),
                author: author.and_then(AuthorJudgement::evaluation),
            })
        })
        .collect()
}

fn pass(
    level: SafetyLevel,
    outer: Option<&Evaluation>,
    inner: &Evaluation,
    author: Option<&AuthorJudgement>,
) -> PassOutcome {
    match outer {
        None => return PassOutcome::OuterAbsent,
        Some(Evaluation::Complete {
            verdict: Verdict::Fosnr(_) | Verdict::Shown { .. },
            ..
        }) => {}
        Some(
            Evaluation::Complete {
                verdict: Verdict::Withheld(_),
                ..
            }
            | Evaluation::Partial { .. }
            | Evaluation::NotFound(_)
            | Evaluation::Deleted(_)
            | Evaluation::Failed(_),
        ) => return PassOutcome::OuterWithheld,
    }
    match (inner, author, judge(level, author)) {
        (
            Evaluation::NotFound(_) | Evaluation::Deleted(_),
            Some(
                AuthorJudgement::Judged(_)
                | AuthorJudgement::Unresolved(Unresolved {
                    cause: Cause::Failed,
                    ..
                }),
            ),
            judged @ (PassOutcome::Dropped(_) | PassOutcome::FailedOpen),
        ) => judged,
        (Evaluation::NotFound(_) | Evaluation::Deleted(_) | Evaluation::Failed(_), _, _) => {
            PassOutcome::InnerUnresolved
        }
        (Evaluation::Complete { .. } | Evaluation::Partial { .. }, _, judged) => judged,
    }
}

fn judge(level: SafetyLevel, author: Option<&AuthorJudgement>) -> PassOutcome {
    let Some(AuthorJudgement::Judged(evaluation)) = author else {
        return PassOutcome::FailedOpen;
    };
    match evaluation {
        Evaluation::Complete {
            verdict: Verdict::Withheld(Decided { id, .. }),
            ..
        } => rerender(level, *id).map_or(PassOutcome::NotRerendered, PassOutcome::Dropped),
        Evaluation::Complete {
            verdict: Verdict::Fosnr(_) | Verdict::Shown { .. },
            ..
        } => PassOutcome::Passed,
        Evaluation::Partial { .. }
        | Evaluation::NotFound(_)
        | Evaluation::Deleted(_)
        | Evaluation::Failed(_) => PassOutcome::FailedOpen,
    }
}

fn rerender(level: SafetyLevel, by: RuleId) -> Option<Rerendered> {
    let QuoteEntries::JudgedWithQuotePass { rerenders } = level.spec().quote_entries() else {
        return None;
    };
    let &(_, reason) = rerenders.iter().find(|(id, _)| *id == by)?;
    Some(Rerendered { by, reason })
}

fn record(metrics: &RequestMetrics, passes: &[QuotePass]) {
    let mut counts: HashMap<&'static str, u64> = HashMap::new();
    for pass in passes {
        *counts.entry((&pass.outcome).into()).or_default() += 1;
    }
    for (outcome, count) in counts {
        metrics.count(&QUOTE_PASS, [outcome], count);
    }
    ft_metrics::record_verdicts(
        metrics,
        passes.iter().filter_map(|pass| match pass.outcome {
            PassOutcome::Dropped(_) | PassOutcome::Passed => {
                pass.author.as_ref().map(Evaluation::verdict)
            }
            PassOutcome::OuterWithheld
            | PassOutcome::OuterAbsent
            | PassOutcome::InnerUnresolved
            | PassOutcome::FailedOpen
            | PassOutcome::NotRerendered => None,
        }),
    );
}

#[cfg(test)]
mod tests;
