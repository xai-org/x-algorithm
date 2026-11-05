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
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeFrom, EdgeQuery, Graph};
    use crate::evaluate_tweets::EntryComparator;
    use crate::hydration::fallback_cache::FallbackCaches;
    use crate::hydration::sources::Source;
    use crate::hydration::sources::{suspended, Fault, InMemorySources};
    use crate::hydration::tweet_fallback_cache;
    use crate::hydration::Relation;
    use crate::metrics::families::PHASE_MS;
    use crate::metrics::{Metrics, Recording};
    use crate::models::{RequestViewer, TweetId};
    use crate::rules::RuleEngine;
    use crate::scenario::{tweet_data, Wire};
    use std::time::Duration;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
    use tokio::time::timeout;
    use vf_pb::tweet_evaluation::Outcome;
    use xai_core_entities::entities::{
        GizmoduckUser, GizmoduckUserResult, Safety, UserResponseState,
    };
    use xai_visibility_filtering_proto as vf_pb;
    use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

    const QUOTE_PAIRS: (Relation, Source) = (Relation::QuoteOuter, Source::Flock);

    const VIEWER: u64 = 50;
    const QUOTED_AUTHOR: u64 = 20;
    const QUOTING_AUTHOR: u64 = 30;
    const INNER: u64 = 1;
    const OUTER: u64 = 9;

    struct Captured(UnboundedSender<Passes>);

    impl EntryComparator for Captured {
        fn compare(
            &self,
            _: &RequestViewer,
            _: SafetyLevel,
            _: &[PostEvaluation],
            quote_passes: &Passes,
        ) {
            self.0.send(quote_passes.clone()).unwrap();
        }
    }

    struct Served {
        level: ThriftLevel,
        wire: Wire,
        compared: UnboundedReceiver<Passes>,
        recording: Arc<Recording>,
    }

    impl Served {
        fn new(
            world: Arc<InMemorySources>,
            caches: FallbackCaches,
            quote_pass: QuotePassRunner,
        ) -> Self {
            let (captured, compared) = unbounded_channel();
            let (metrics, recording) = Metrics::recording();
            let wire = Wire::serving(
                world,
                caches,
                Some(Box::new(Captured(captured))),
                quote_pass,
                metrics,
            );
            Self {
                level: ThriftLevel::TIMELINE_HOME_HYDRATION,
                wire,
                compared,
                recording,
            }
        }

        async fn outcomes(
            &self,
            viewer_id: Option<u64>,
            tweets: Vec<vf_pb::TweetData>,
        ) -> Vec<Outcome> {
            self.wire
                .send(vf_pb::EvaluateTweetsRequest {
                    safety_level: self.level.0,
                    viewer_id,
                    tweets,
                })
                .await
                .unwrap()
        }

        async fn compared(&mut self) -> Passes {
            timeout(Duration::from_secs(5), self.compared.recv())
                .await
                .expect("the comparator hears of every request")
                .unwrap()
        }

        async fn serve(
            &mut self,
            viewer_id: Option<u64>,
            tweets: Vec<vf_pb::TweetData>,
        ) -> (Vec<Outcome>, Vec<QuotePass>) {
            let outcomes = self.outcomes(viewer_id, tweets).await;
            (outcomes, self.compared().await.ran().to_vec())
        }
    }

    fn author(safety: Safety) -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety,
                ..Default::default()
            }),
            response_state: Some(UserResponseState::Found),
        }
    }

    fn protected() -> GizmoduckUserResult {
        author(Safety {
            is_protected: true,
            ..Default::default()
        })
    }

    fn world() -> InMemorySources {
        InMemorySources::default()
            .tweet(INNER, QUOTED_AUTHOR)
            .tweet(OUTER, QUOTING_AUTHOR)
            .authors(&[QUOTED_AUTHOR, QUOTING_AUTHOR])
    }

    fn deleted_inner() -> InMemorySources {
        InMemorySources::default()
            .tweet(OUTER, QUOTING_AUTHOR)
            .quote(OUTER, INNER, QUOTED_AUTHOR)
            .authors(&[QUOTED_AUTHOR, QUOTING_AUTHOR])
    }

    fn live_inner_of_a_not_found_author() -> InMemorySources {
        InMemorySources::default()
            .tweet(INNER, QUOTED_AUTHOR)
            .tweet(OUTER, QUOTING_AUTHOR)
            .authors(&[QUOTING_AUTHOR])
    }

    fn quote_with_outer() -> Vec<vf_pb::TweetData> {
        vec![tweet_data(OUTER, None), tweet_data(INNER, Some(OUTER))]
    }

    async fn serve(
        world: Arc<InMemorySources>,
        mode: QuotePassMode,
        viewer_id: Option<u64>,
        tweets: Vec<vf_pb::TweetData>,
    ) -> (Vec<Outcome>, Vec<QuotePass>) {
        Served::new(world, FallbackCaches::default(), QuotePassRunner::new(mode))
            .serve(viewer_id, tweets)
            .await
    }

    fn one_permit() -> QuotePassRunner {
        QuotePassRunner {
            mode: QuotePassMode::Shadow,
            permits: Arc::new(Semaphore::new(1)),
        }
    }

    fn dropped(rule: &str, reason: RerenderReason) -> PassOutcome {
        PassOutcome::Dropped(Rerendered {
            by: RuleId::of_rule_name(rule).unwrap(),
            reason,
        })
    }

    fn erased_drop() -> PassOutcome {
        dropped(
            "erased_author/drop/inactive",
            RerenderReason::AuthorDeactivated,
        )
    }

    struct PassCase {
        name: &'static str,
        viewer_id: Option<u64>,
        world: InMemorySources,
        tweets: Vec<vf_pb::TweetData>,
        outcome: PassOutcome,
    }

    fn case(name: &'static str, world: InMemorySources, outcome: PassOutcome) -> PassCase {
        PassCase {
            name,
            viewer_id: Some(VIEWER),
            world,
            tweets: quote_with_outer(),
            outcome,
        }
    }

    fn pass_cases() -> Vec<PassCase> {
        let deactivated = Safety {
            deactivated: true,
            ..Default::default()
        };
        let erased = Safety {
            erased: true,
            ..Default::default()
        };
        let offboarded = Safety {
            offboarded: true,
            ..Default::default()
        };
        vec![
            case("passed", world(), PassOutcome::Passed),
            case(
                "deactivated",
                world().user(QUOTED_AUTHOR, author(deactivated.clone())),
                dropped("deactivated_author/drop", RerenderReason::AuthorDeactivated),
            ),
            case(
                "erased",
                world().user(QUOTED_AUTHOR, author(erased)),
                dropped(
                    "erased_author/drop/inactive",
                    RerenderReason::AuthorDeactivated,
                ),
            ),
            case(
                "offboarded",
                world().user(QUOTED_AUTHOR, author(offboarded)),
                dropped(
                    "offboarded_author/drop/inactive",
                    RerenderReason::AuthorInactive,
                ),
            ),
            case(
                "suspended",
                world().user(QUOTED_AUTHOR, suspended()),
                dropped("suspended_author/drop", RerenderReason::AuthorSuspended),
            ),
            case(
                "quoted author blocks quoting author",
                world().edge(Graph::Blocks, QUOTED_AUTHOR, QUOTING_AUTHOR),
                dropped(
                    "quoted_author_blocks_quoting_author/drop/author_block_viewer",
                    RerenderReason::AuthorBlocksViewer,
                ),
            ),
            case(
                "viewer blocks author",
                world().edge(Graph::Blocks, VIEWER, QUOTED_AUTHOR),
                dropped("viewer_blocks_author/drop", RerenderReason::Unspecified),
            ),
            case(
                "viewer mutes author",
                world().edge(Graph::Mutes, VIEWER, QUOTED_AUTHOR),
                dropped(
                    "quoted_viewer_mutes_author_not_following/drop",
                    RerenderReason::Unspecified,
                ),
            ),
            case(
                "protected",
                world().user(QUOTED_AUTHOR, protected()),
                dropped(
                    "protected_quoted_author/drop",
                    RerenderReason::AuthorProtected,
                ),
            ),
            case(
                "outer withheld",
                world().user(QUOTING_AUTHOR, suspended()),
                PassOutcome::OuterWithheld,
            ),
            case(
                "outer not found",
                InMemorySources::default()
                    .tweet(INNER, QUOTED_AUTHOR)
                    .authors(&[QUOTED_AUTHOR]),
                PassOutcome::OuterWithheld,
            ),
            case(
                "outer failed",
                world().fail_key(Source::TesPureCoreData, OUTER),
                PassOutcome::OuterWithheld,
            ),
            PassCase {
                tweets: vec![tweet_data(INNER, Some(OUTER))],
                ..case(
                    "outer absent",
                    world().user(QUOTED_AUTHOR, suspended()),
                    PassOutcome::OuterAbsent,
                )
            },
            case(
                "inner deleted",
                InMemorySources::default()
                    .tweet(OUTER, QUOTING_AUTHOR)
                    .authors(&[QUOTING_AUTHOR]),
                PassOutcome::InnerUnresolved,
            ),
            case(
                "deleted inner whose author blocks the quoting author",
                deleted_inner().edge(Graph::Blocks, QUOTED_AUTHOR, QUOTING_AUTHOR),
                dropped(
                    "quoted_author_blocks_quoting_author/drop/author_block_viewer",
                    RerenderReason::AuthorBlocksViewer,
                ),
            ),
            case(
                "deleted inner of a deactivated author",
                deleted_inner().user(QUOTED_AUTHOR, author(deactivated)),
                dropped("deactivated_author/drop", RerenderReason::AuthorDeactivated),
            ),
            case(
                "deleted inner of a protected author",
                deleted_inner().user(QUOTED_AUTHOR, protected()),
                dropped(
                    "protected_quoted_author/drop",
                    RerenderReason::AuthorProtected,
                ),
            ),
            case(
                "deleted inner whose author level 65 does not drop",
                deleted_inner(),
                PassOutcome::InnerUnresolved,
            ),
            case(
                "deleted inner of a not-found author",
                InMemorySources::default()
                    .tweet(OUTER, QUOTING_AUTHOR)
                    .quote(OUTER, INNER, QUOTED_AUTHOR)
                    .authors(&[QUOTING_AUTHOR]),
                erased_drop(),
            ),
            case(
                "live inner of a not-found author",
                live_inner_of_a_not_found_author(),
                erased_drop(),
            ),
            case(
                "live inner of a soft user",
                world().user(
                    QUOTED_AUTHOR,
                    GizmoduckUserResult {
                        user: None,
                        response_state: Some(UserResponseState::SoftUser),
                    },
                ),
                erased_drop(),
            ),
            case(
                "deleted inner whose author lookup failed",
                deleted_inner().fail_key(Source::GizmoduckAuthor, QUOTED_AUTHOR),
                PassOutcome::FailedOpen,
            ),
            case(
                "live inner whose author lookup failed",
                world().fail_key(Source::GizmoduckAuthor, QUOTED_AUTHOR),
                PassOutcome::InnerUnresolved,
            ),
            case(
                "deleted inner whose outer's quoted tweet lookup failed",
                deleted_inner()
                    .edge(Graph::Blocks, QUOTED_AUTHOR, QUOTING_AUTHOR)
                    .fault(Source::TesQuotedTweet, Fault::Fails),
                PassOutcome::FailedOpen,
            ),
            case(
                "block pair edge failed",
                world()
                    .user(QUOTED_AUTHOR, protected())
                    .edge(Graph::Follows, QUOTING_AUTHOR, QUOTED_AUTHOR)
                    .fault(QUOTE_PAIRS, Fault::Fails),
                PassOutcome::FailedOpen,
            ),
            PassCase {
                viewer_id: None,
                ..case(
                    "follow pair edge failed for a protected author the quoting author follows",
                    world()
                        .user(QUOTED_AUTHOR, protected())
                        .edge(Graph::Follows, QUOTING_AUTHOR, QUOTED_AUTHOR)
                        .fail_graph(Graph::Follows),
                    PassOutcome::FailedOpen,
                )
            },
        ]
    }

    #[tokio::test]
    async fn the_shadow_pass_judges_outer_then_quoted_author_by_every_wired_rule_serving_as_off() {
        let quote = || EvaluationKey {
            tweet_id: TweetId(INNER),
            placement: Placement::InnerQuoted {
                outer: TweetId(OUTER),
            },
        };
        let mut rerendered = vec![];
        for PassCase {
            name,
            viewer_id,
            world,
            tweets,
            outcome,
        } in pass_cases()
        {
            let world = Arc::new(world);
            let (off, no_passes) = serve(
                Arc::clone(&world),
                QuotePassMode::Off,
                viewer_id,
                tweets.clone(),
            )
            .await;
            assert!(no_passes.is_empty(), "{name}");
            assert!(world.starts(QUOTE_PAIRS).is_empty(), "{name}");
            assert!(!world.calls().contains(&Source::TesQuotedTweet), "{name}");
            let (shadow, passes) = serve(world, QuotePassMode::Shadow, viewer_id, tweets).await;
            assert_eq!(
                shadow, off,
                "{name}: shadow serves the inner-quoted verdict"
            );
            assert_eq!(
                passes
                    .iter()
                    .map(|pass| (pass.key, &pass.outcome))
                    .collect::<Vec<_>>(),
                [(quote(), &outcome)],
                "{name}"
            );
            if let PassOutcome::Dropped(Rerendered { by, .. }) = outcome {
                rerendered.push(by);
            }
        }
        let undecided: Vec<&str> = RuleEngine::for_tests()
            .wired_rule_names(SafetyLevel::QuotedTweetRules)
            .into_iter()
            .filter(|&rule| !rerendered.contains(&RuleId::of_rule_name(rule).unwrap()))
            .collect();
        assert!(
            undecided.is_empty(),
            "quoted_tweet_rules rules no quote entry rerenders: {undecided:?}"
        );
    }

    #[tokio::test]
    async fn a_level_124_quote_entry_keeps_its_own_verdict_and_runs_no_pass() {
        let world = Arc::new(world().user(QUOTED_AUTHOR, protected()));
        let mut served = Served::new(
            Arc::clone(&world),
            FallbackCaches::default(),
            QuotePassRunner::new(QuotePassMode::Shadow),
        );
        served.level = ThriftLevel::TIMELINE_HOME_PROMOTED_HYDRATION;
        let outcomes = served.outcomes(Some(VIEWER), quote_with_outer()).await;
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome, Outcome::ResultStateThriftCompact(_))),
            "{outcomes:?}"
        );
        assert_eq!(served.compared().await, Passes::Skipped);
        assert!(world.starts(QUOTE_PAIRS).is_empty());
    }

    #[test]
    fn a_level_65_drop_rerenders_by_its_rule_identity_whatever_its_name() {
        use crate::models::{DropReason, Withholding};
        use xai_visibility_filtering::models::FilteredReason;
        let author = AuthorJudgement::Judged(Evaluation::complete(Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
            id: RuleId::SuspendedAuthor,
            by: "renamed",
        })));
        assert!(matches!(
            judge(SafetyLevel::TimelineHomeHydration, Some(&author)),
            PassOutcome::Dropped(Rerendered {
                by: RuleId::SuspendedAuthor,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn a_not_found_quoted_author_sends_not_found_for_the_served_inner() {
        use xai_x_thrift::tweet_service::{TweetFieldsResultNotFound, TweetFieldsResultState};
        let (served, _) = serve(
            Arc::new(live_inner_of_a_not_found_author()),
            QuotePassMode::Off,
            Some(VIEWER),
            quote_with_outer(),
        )
        .await;
        assert!(
            matches!(served[0], Outcome::ResultStateThriftCompact(_)),
            "{served:?}"
        );
        let not_found =
            TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(false, false, None));
        assert_eq!(
            served[1],
            Outcome::ResultStateThriftCompact(
                xai_x_thrift::serialize_compact(&not_found).unwrap().into()
            )
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_deleted_inner_drops_for_a_protected_author_a_live_quote_already_hydrated() {
        const LIVE_INNER: u64 = 2;
        const LIVE_OUTER: u64 = 8;
        const LIVE_QUOTING_AUTHOR: u64 = 31;
        let world = deleted_inner()
            .tweet(LIVE_INNER, QUOTED_AUTHOR)
            .tweet(LIVE_OUTER, LIVE_QUOTING_AUTHOR)
            .authors(&[LIVE_QUOTING_AUTHOR])
            .user(QUOTED_AUTHOR, protected())
            .latency(Source::TesQuotedTweet, Duration::from_millis(10));
        let mut tweets = quote_with_outer();
        tweets.extend([
            tweet_data(LIVE_OUTER, None),
            tweet_data(LIVE_INNER, Some(LIVE_OUTER)),
        ]);
        let (_, passes) = serve(Arc::new(world), QuotePassMode::Shadow, Some(VIEWER), tweets).await;
        let protected_drop = dropped(
            "protected_quoted_author/drop",
            RerenderReason::AuthorProtected,
        );
        assert_eq!(
            passes
                .iter()
                .map(|pass| (pass.key.tweet_id.0, &pass.outcome))
                .collect::<Vec<_>>(),
            [(INNER, &protected_drop), (LIVE_INNER, &protected_drop)]
        );
    }

    #[tokio::test]
    async fn a_viewer_who_quotes_a_protected_author_asks_the_follow_as_viewer_and_as_quoting_author(
    ) {
        let world = Arc::new(world().user(QUOTED_AUTHOR, protected()).edge(
            Graph::Follows,
            QUOTING_AUTHOR,
            QUOTED_AUTHOR,
        ));
        let (_, passes) = serve(
            Arc::clone(&world),
            QuotePassMode::Shadow,
            Some(QUOTING_AUTHOR),
            quote_with_outer(),
        )
        .await;
        assert_eq!(
            passes.iter().map(|pass| &pass.outcome).collect::<Vec<_>>(),
            [&PassOutcome::Passed]
        );
        let follows_quoted_author = |from| EdgeQuery {
            from,
            ..EdgeQuery::forward(Graph::Follows, vec![QUOTED_AUTHOR])
        };
        let queries: Vec<EdgeQuery> = world.selects().into_iter().flatten().collect();
        assert!(queries.contains(&follows_quoted_author(EdgeFrom::Viewer)));
        assert!(queries.contains(&follows_quoted_author(EdgeFrom::User(QUOTING_AUTHOR))));
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_quote_pass_source_does_not_delay_the_served_response() {
        let world = Arc::new(world().fault(QUOTE_PAIRS, Fault::Hangs));
        let (off, _) = serve(
            Arc::clone(&world),
            QuotePassMode::Off,
            Some(VIEWER),
            quote_with_outer(),
        )
        .await;
        let mut shadow = Served::new(
            world,
            FallbackCaches::default(),
            QuotePassRunner::new(QuotePassMode::Shadow),
        );
        let started = Instant::now();
        let served = shadow.outcomes(Some(VIEWER), quote_with_outer()).await;
        assert_eq!(started.elapsed(), Duration::ZERO);
        assert_eq!(served, off);
        let passes = shadow.compared().await.ran().to_vec();
        assert!(
            started.elapsed() > Duration::ZERO,
            "the pass waited out the hung source"
        );
        assert_eq!(
            passes.iter().map(|pass| &pass.outcome).collect::<Vec<_>>(),
            [&PassOutcome::FailedOpen]
        );
    }

    #[tokio::test]
    async fn a_finished_pass_frees_its_permit_for_the_next_request() {
        let mut served = Served::new(Arc::new(world()), FallbackCaches::default(), one_permit());
        for request in ["first", "second"] {
            let (_, passes) = served.serve(Some(VIEWER), quote_with_outer()).await;
            assert_eq!(
                passes.iter().map(|pass| &pass.outcome).collect::<Vec<_>>(),
                [&PassOutcome::Passed],
                "{request}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_running_pass_holds_its_permit_until_it_ends() {
        let world = Arc::new(world().fault(QUOTE_PAIRS, Fault::Hangs));
        let mut served = Served::new(Arc::clone(&world), FallbackCaches::default(), one_permit());
        served.outcomes(Some(VIEWER), quote_with_outer()).await;
        served.outcomes(Some(VIEWER), quote_with_outer()).await;
        assert_eq!(
            served.compared().await,
            Passes::Shed,
            "the second pass is shed"
        );
        assert_eq!(
            served
                .compared()
                .await
                .ran()
                .iter()
                .map(|pass| &pass.outcome)
                .collect::<Vec<_>>(),
            [&PassOutcome::FailedOpen]
        );
        assert_eq!(
            world.keys(QUOTE_PAIRS).len(),
            1,
            "the shed pass fetches nothing"
        );
        let quote_pass = served.recording.take(&QUOTE_PASS.family);
        assert!(
            quote_pass.iter().all(|sample| {
                sample.label("rpc") == Some("evaluate_tweets")
                    && sample.label("safety_level") == Some("quoted_tweet_rules")
            }),
            "{quote_pass:?}"
        );
        let shed = quote_pass
            .iter()
            .filter(|sample| sample.label("outcome") == Some(SHED))
            .map(|sample| sample.value)
            .sum::<f64>();
        assert_eq!(shed, 1.0);
    }

    #[tokio::test]
    async fn the_quote_pass_fills_no_fallback_cache_the_served_hydration_reads() {
        let world = Arc::new(world());
        let caches = FallbackCaches {
            tweets: Some(tweet_fallback_cache(8)),
            authors: None,
        };
        let mut served = Served::new(
            Arc::clone(&world),
            caches,
            QuotePassRunner::new(QuotePassMode::Shadow),
        );
        let (_, passes) = served
            .serve(Some(VIEWER), vec![tweet_data(INNER, Some(OUTER))])
            .await;
        assert_eq!(passes[0].outcome, PassOutcome::OuterAbsent);
        assert!(
            world
                .keys(Source::TesPureCoreData)
                .concat()
                .contains(&OUTER),
            "the pass fetched the outer's pure core"
        );
        world.break_source(Source::TesPureCoreData, Fault::Fails);
        let (outer, _) = served
            .serve(Some(VIEWER), vec![tweet_data(OUTER, None)])
            .await;
        assert_eq!(outer, [Outcome::Failed(vf_pb::Failed {})]);
    }

    #[tokio::test]
    async fn a_shadow_request_records_one_hydration_phase_sample() {
        let mut served = Served::new(
            Arc::new(world()),
            FallbackCaches::default(),
            QuotePassRunner::new(QuotePassMode::Shadow),
        );
        let (_, passes) = served.serve(Some(VIEWER), quote_with_outer()).await;
        let hydrations: Vec<Option<String>> = served
            .recording
            .take(&PHASE_MS.family)
            .iter()
            .filter(|sample| sample.label("stage") == Some("hydration"))
            .map(|sample| sample.label("rpc").map(str::to_owned))
            .collect();
        assert_eq!(passes.len(), 1, "the pass ran");
        assert_eq!(hydrations, [Some("evaluate_tweets".to_owned())]);
    }
}
