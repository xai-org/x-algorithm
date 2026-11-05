use super::{PostBatch, PostEvaluator};
use crate::hydration::fallback_cache::FallbackCaches;
use crate::hydration::{
    Cause, HydratedTweet, HydratedUser, HydrationRequest, Hydrator, Hydrators, Lookup, Unresolved,
};
use crate::metrics::RequestMetrics;
use crate::models::{AuthorId, Evaluation, EvaluationKey, RequestFacts};
use crate::rules::{LevelSpec, SafetyLevel};
use std::collections::HashMap;
use tracing::warn;

pub(super) const VIEWER_LOOKUPS: Hydrators = Hydrators::of(Hydrator::ViewerProfile)
    .with(Hydrator::ViewerLabels)
    .with(Hydrator::ViewerNotificationFilters);

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AuthorJudgement {
    Judged(Evaluation),
    Unresolved(Unresolved),
}

impl AuthorJudgement {
    fn judged(evaluation: Evaluation, failed: Hydrators) -> Self {
        Self::Judged(match evaluation {
            Evaluation::Complete {
                verdict,
                fail_open_defaults,
            } if !failed.is_empty() => Evaluation::Partial {
                verdict,
                fail_open_defaults,
            },
            evaluation @ (Evaluation::Complete { .. }
            | Evaluation::Partial { .. }
            | Evaluation::NotFound(_)
            | Evaluation::Deleted(_)
            | Evaluation::Failed(_)) => evaluation,
        })
    }

    pub(crate) fn evaluation(&self) -> Option<Evaluation> {
        match self {
            Self::Judged(evaluation) => Some(evaluation.clone()),
            Self::Unresolved(Unresolved {
                lookup,
                cause: Cause::Failed,
            }) => Some(Evaluation::Failed(*lookup)),
            Self::Unresolved(Unresolved {
                cause: Cause::NotFound,
                ..
            }) => None,
        }
    }
}

impl PostEvaluator {
    pub(crate) async fn evaluate_authors(
        &self,
        request: PostBatch,
        metrics: &RequestMetrics,
    ) -> HashMap<EvaluationKey, AuthorJudgement> {
        let level = request.safety_level;
        match level.spec() {
            LevelSpec::QuotePass { .. } => {}
            LevelSpec::Posts(_) | LevelSpec::Notifications { .. } => {
                warn!(
                    ?level,
                    "authors evaluated outside the quote pass level; failing them"
                );
                let failed = AuthorJudgement::Unresolved(Unresolved {
                    lookup: Lookup::Tweet,
                    cause: Cause::Failed,
                });
                return request
                    .candidates
                    .iter()
                    .map(|placed| (placed.key(), failed.clone()))
                    .collect();
            }
        }
        let hydration = self
            .rule_engine
            .plan(level)
            .hydrate(
                &*self.sources,
                request.hydration_request(&FallbackCaches::default(), metrics),
            )
            .await;
        let admitted = self.rule_engine.admit(level, &hydration);
        request
            .candidates
            .iter()
            .map(|placed| {
                let judgement = match hydration.tweet(placed.tweet_id()) {
                    Some(HydratedTweet::Unresolved { reason, .. }) => {
                        AuthorJudgement::Unresolved(*reason)
                    }
                    Some(HydratedTweet::Resolved { .. }) | None => admitted
                        .as_ref()
                        .and_then(|admitted| admitted.author(placed.tweet_id(), placed.placement))
                        .map_or_else(
                            || {
                                AuthorJudgement::Unresolved(Unresolved {
                                    lookup: Lookup::Tweet,
                                    cause: Cause::Failed,
                                })
                            },
                            |author| {
                                AuthorJudgement::judged(
                                    author.evaluate(),
                                    author.user_candidate().failed,
                                )
                            },
                        ),
                };
                (placed.key(), judgement)
            })
            .collect()
    }

    pub(crate) async fn evaluate_users(
        &self,
        level: SafetyLevel,
        facts: RequestFacts,
        users: Vec<AuthorId>,
        metrics: &RequestMetrics,
    ) -> HashMap<AuthorId, Option<Evaluation>> {
        match level.spec() {
            LevelSpec::Notifications { .. } => {}
            LevelSpec::Posts(_) | LevelSpec::QuotePass { .. } => {
                warn!(
                    ?level,
                    "users evaluated outside a notification level; failing them"
                );
                return users.into_iter().map(|id| (id, None)).collect();
            }
        }
        let hydration = self
            .rule_engine
            .plan(level)
            .hydrate(
                &*self.sources,
                HydrationRequest::for_users(facts, users.clone(), &self.fallback_caches, metrics),
            )
            .await;
        let viewer_failed = !hydration
            .viewer_failed()
            .intersection(VIEWER_LOOKUPS)
            .is_empty();
        let admitted = self.rule_engine.admit(level, &hydration);
        users
            .into_iter()
            .map(|id| {
                let result = match (&admitted, hydration.user(id)) {
                    _ if viewer_failed => None,
                    (_, Some(HydratedUser::Unresolved(Cause::NotFound))) => {
                        Some(Evaluation::NotFound(Lookup::Author))
                    }
                    (Some(admitted), _) => admitted.user(id).map(|user| user.evaluate()),
                    (None, _) => None,
                };
                (id, result)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeFrom, EdgeQuery, Graph};
    use crate::hydration::sources::InMemorySources;
    use crate::hydration::sources::Source;
    use crate::hydration::Hydrator;
    use crate::metrics::{Endpoint, Metrics, RequestLevel};
    use crate::models::{PlacedCandidate, Placement, RequestFacts, RetweetMerge, TweetId};
    use crate::rules::fixtures::{allow, dropped};
    use crate::rules::{RuleEngine, SafetyLevel};
    use crate::scenario::raw;
    use std::sync::Arc;
    use xai_core_entities::entities::{
        GizmoduckUser, GizmoduckUserResult, Safety, UserResponseState,
    };
    use xai_visibility_filtering::models::FilteredReason;

    #[tokio::test]
    async fn the_quote_pass_fetches_a_missing_outer_and_reads_its_pair_edges_logged_out() {
        const QUOTING_AUTHOR: u64 = 30;
        let protected = || GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety: Safety {
                    is_protected: true,
                    ..Default::default()
                },
                ..Default::default()
            }),
            response_state: Some(UserResponseState::Found),
        };
        let sources = Arc::new(
            InMemorySources::default()
                .tweet(1, 20)
                .tweet(2, 21)
                .tweet(3, 22)
                .tweet(4, QUOTING_AUTHOR)
                .tweet(9, QUOTING_AUTHOR)
                .authors(&[20, QUOTING_AUTHOR])
                .user(21, protected())
                .user(22, protected())
                .user(QUOTING_AUTHOR, protected())
                .edge(Graph::Blocks, 20, QUOTING_AUTHOR)
                .edge(Graph::Follows, QUOTING_AUTHOR, 22),
        );
        let post_evaluator = PostEvaluator::new(
            Arc::clone(&sources) as _,
            RuleEngine::for_tests(),
            Arc::default(),
            Metrics::none(),
        );
        let quoted = |tweet_id| PlacedCandidate {
            candidate: raw(tweet_id, None),
            placement: Placement::InnerQuoted { outer: TweetId(9) },
        };
        let authors = post_evaluator
            .evaluate_authors(
                PostBatch {
                    facts: RequestFacts::default(),
                    safety_level: SafetyLevel::QuotedTweetRules,
                    candidates: [1, 2, 3, 4].map(quoted).into(),
                    retweet_merge: RetweetMerge::Unmerged,
                },
                &Metrics::none().request(
                    Endpoint::EvaluateTweets,
                    RequestLevel::Named(SafetyLevel::QuotedTweetRules),
                ),
            )
            .await;
        let judged = |verdict| AuthorJudgement::Judged(Evaluation::complete(verdict));
        assert_eq!(
            authors,
            HashMap::from([
                (
                    quoted(1).key(),
                    judged(dropped(
                        FilteredReason::AuthorBlockViewer,
                        "quoted_author_blocks_quoting_author/drop/author_block_viewer"
                    ))
                ),
                (
                    quoted(2).key(),
                    judged(dropped(
                        FilteredReason::AuthorIsProtected,
                        "protected_quoted_author/drop"
                    ))
                ),
                (quoted(3).key(), judged(allow())),
                (quoted(4).key(), judged(allow())),
            ])
        );
        assert_eq!(
            sources.keys(Source::TesPureCoreData),
            [vec![1, 2, 3, 4], vec![9]]
        );
        let sourced = |graph, source_id, destination_ids| EdgeQuery {
            from: EdgeFrom::User(source_id),
            ..EdgeQuery::forward(graph, destination_ids)
        };
        let mut selects = sources.selects();
        selects.sort_by_key(|queries| queries.first().map(|query| <&str>::from(query.graph)));
        assert_eq!(
            selects,
            [
                vec![
                    sourced(Graph::Blocks, 20, vec![QUOTING_AUTHOR]),
                    sourced(Graph::Blocks, 21, vec![QUOTING_AUTHOR]),
                    sourced(Graph::Blocks, 22, vec![QUOTING_AUTHOR]),
                ],
                vec![sourced(Graph::Follows, QUOTING_AUTHOR, vec![21, 22])],
            ]
        );
    }

    #[tokio::test]
    async fn a_failed_outer_fails_only_its_quotes_pair_nodes() {
        let sources = InMemorySources::default()
            .tweet(1, 20)
            .tweet(2, 20)
            .tweet(8, 30)
            .authors(&[20, 30])
            .fail_key(Source::TesPureCoreData, 9);
        let post_evaluator = PostEvaluator::new(
            Arc::new(sources),
            RuleEngine::for_tests(),
            Arc::default(),
            Metrics::none(),
        );
        let quoted = |tweet_id, outer| PlacedCandidate {
            candidate: raw(tweet_id, None),
            placement: Placement::InnerQuoted {
                outer: TweetId(outer),
            },
        };
        let batch = || PostBatch {
            facts: RequestFacts::logged_in(50),
            safety_level: SafetyLevel::QuotedTweetRules,
            candidates: vec![quoted(1, 8), quoted(2, 9)],
            retweet_merge: RetweetMerge::Unmerged,
        };
        let authors = post_evaluator
            .evaluate_authors(
                batch(),
                &Metrics::none().request(
                    Endpoint::EvaluateTweets,
                    RequestLevel::Named(SafetyLevel::QuotedTweetRules),
                ),
            )
            .await;
        assert_eq!(
            authors[&quoted(1, 8).key()],
            AuthorJudgement::Judged(Evaluation::complete(allow()))
        );
        assert_eq!(
            authors[&quoted(2, 9).key()],
            AuthorJudgement::Judged(Evaluation::Partial {
                verdict: allow(),
                fail_open_defaults: Hydrators::of(Hydrator::AuthorBlocksOuterAuthor),
            })
        );
        let hydration = post_evaluator
            .rule_engine
            .plan(SafetyLevel::QuotedTweetRules)
            .hydrate(
                &*post_evaluator.sources,
                batch().hydration_request(
                    &FallbackCaches::default(),
                    &Metrics::none().request(
                        Endpoint::EvaluateTweets,
                        RequestLevel::Named(SafetyLevel::QuotedTweetRules),
                    ),
                ),
            )
            .await;
        let admitted = post_evaluator
            .rule_engine
            .admit(SafetyLevel::QuotedTweetRules, &hydration)
            .unwrap();
        let author = admitted.author(TweetId(2), quoted(2, 9).placement).unwrap();
        assert_eq!(
            author.user_candidate().failed,
            Hydrators::of(Hydrator::AuthorBlocksOuterAuthor)
                .with(Hydrator::OuterAuthorFollowsAuthor)
        );
    }
}
