use crate::ingress::{self, Entry};
use crate::metrics::{self as ft_metrics, Endpoint, RequestLevel, RequestMetrics};
use crate::models::{Evaluation, RequestViewer};
use crate::params::ClientSwitches;
use crate::post_evaluator::{PostEvaluation, PostEvaluator};
use crate::quote::{Passes, QuotePassRunner};
use crate::rules::{SafetyLevel, WireLevel};
use crate::treatment;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use vf_pb::tweet_evaluation::Outcome;
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) trait EntryComparator: Send + Sync {
    fn compare(
        &self,
        viewer: &RequestViewer,
        safety_level: SafetyLevel,
        outcomes: &[PostEvaluation],
        quote_passes: &Passes,
    );
}

pub struct EvaluateTweetsEndpoint {
    post_evaluator: Arc<PostEvaluator>,
    client_switches: ClientSwitches,
    comparator: Option<Arc<dyn EntryComparator>>,
    quote_pass: QuotePassRunner,
}

impl EvaluateTweetsEndpoint {
    pub(crate) fn new(
        post_evaluator: Arc<PostEvaluator>,
        client_switches: ClientSwitches,
        comparator: Option<Box<dyn EntryComparator>>,
        quote_pass: QuotePassRunner,
    ) -> Self {
        Self {
            post_evaluator,
            client_switches,
            comparator: comparator.map(Arc::from),
            quote_pass,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::EvaluateTweetsRequest>,
    ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
        let safety_level =
            ingress::level(WireLevel::EvaluateTweets(request.get_ref().safety_level));
        ft_metrics::observe_request(
            self.post_evaluator.metrics(),
            Endpoint::EvaluateTweets,
            RequestLevel::of(safety_level.as_ref().ok().copied()),
            request,
            |request, context, metrics| async move {
                let twitter_context =
                    xai_twittercontext::extract_twitter_context(request.metadata());
                context
                    .scope(self.handle_inner(
                        request.into_inner(),
                        twitter_context,
                        &metrics,
                        safety_level,
                    ))
                    .await
                    .map(Response::new)
            },
        )
        .await
    }

    async fn handle_inner(
        &self,
        req: vf_pb::EvaluateTweetsRequest,
        twitter_context: Option<TwitterContextViewer>,
        metrics: &RequestMetrics,
        safety_level: Result<SafetyLevel, Status>,
    ) -> Result<vf_pb::EvaluateTweetsResponse, Status> {
        ft_metrics::record_batch_size(metrics, req.tweets.len());
        let safety_level = safety_level?;
        let ingress::Request {
            viewer,
            rows: entries,
        } = ingress::evaluate_tweets(&req, twitter_context, safety_level);
        let candidates = entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Judged(candidate) => Some(*candidate),
                Entry::NotJudged => None,
            })
            .collect();
        let request = ingress::evaluate_tweets_batch(
            &self.client_switches,
            viewer.clone(),
            safety_level,
            candidates,
        );
        let (outcomes, pending) = self
            .quote_pass
            .serve(&self.post_evaluator, request, metrics)
            .await;
        let mut evaluated = outcomes.iter().peekable();
        let answers: Vec<Answer> = entries
            .iter()
            .map(|entry| match entry {
                Entry::NotJudged => Answer::NotJudged,
                Entry::Judged(candidate) => {
                    match evaluated.next_if(|outcome| outcome.key() == candidate.key()) {
                        Some(outcome) => Answer::Evaluated(&outcome.evaluation),
                        None => Answer::Unpaired,
                    }
                }
            })
            .collect();
        let evaluations = || {
            answers.iter().filter_map(|answer| match answer {
                Answer::Evaluated(evaluation) => Some(*evaluation),
                Answer::NotJudged | Answer::Unpaired => None,
            })
        };
        ft_metrics::record_sent(
            metrics,
            evaluations(),
            answers
                .iter()
                .filter(|answer| matches!(answer, Answer::Unpaired))
                .count(),
        );
        let policies = self.client_switches.limited_actions_policies(
            viewer.context(),
            viewer.id(),
            viewer.country_code(),
            evaluations().flat_map(|evaluation| evaluation.verdict().limited_engagement_reasons()),
            metrics.plain(),
        );
        let results = answers
            .iter()
            .map(|answer| {
                let outcome = match answer {
                    Answer::NotJudged => Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                    Answer::Evaluated(evaluation) => treatment::evaluate_tweets_outcome(
                        Some(evaluation),
                        safety_level,
                        &policies,
                    ),
                    Answer::Unpaired => {
                        treatment::evaluate_tweets_outcome(None, safety_level, &policies)
                    }
                };
                vf_pb::TweetEvaluation {
                    outcome: Some(outcome),
                }
            })
            .collect();
        let comparator = self.comparator.clone();
        self.quote_pass.finish(
            &self.post_evaluator,
            pending,
            outcomes,
            move |outcomes, quote_passes| {
                if let Some(comparator) = comparator {
                    comparator.compare(&viewer, safety_level, outcomes, quote_passes);
                }
            },
        );
        Ok(vf_pb::EvaluateTweetsResponse { results })
    }
}

enum Answer<'a> {
    NotJudged,
    Unpaired,
    Evaluated(&'a Evaluation),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::sources::Source;
    use crate::hydration::sources::{Fault, InMemorySources};
    use crate::metrics::Metrics;
    use crate::models::{
        ClientSwitch, PlacedCandidate, RawCandidate, RetweetMerge, TweetFeatures, TweetId,
    };
    use crate::post_evaluator::PostBatch;
    use crate::rules::{NoTrace, RuleEngine};
    use crate::scenario::{tweet_data, Wire};
    use std::collections::HashMap;
    use xai_core_entities::entities::{
        GizmoduckUser, GizmoduckUserResult, PureCoreData, Safety, UserResponseState,
    };
    use xai_x_thrift::action::{self, Action, DropReason};
    use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;
    use xai_x_thrift::safety_result::{FilteredReason as ThriftFilteredReason, SafetyResult};
    use xai_x_thrift::tweet_service::{
        TweetFieldsResultFiltered, TweetFieldsResultFound, TweetFieldsResultNotFound,
        TweetFieldsResultState,
    };

    fn encoded(state: &TweetFieldsResultState) -> Outcome {
        Outcome::ResultStateThriftCompact(xai_x_thrift::serialize_compact(state).unwrap().into())
    }

    fn filtered(reason: ThriftFilteredReason) -> Outcome {
        encoded(&TweetFieldsResultState::Filtered(
            TweetFieldsResultFiltered::new(reason),
        ))
    }

    #[tokio::test]
    async fn evaluate_tweets_gates_levels_maps_outcomes_and_merges_retweet_sources() {
        let core = |author_id, source_tweet_id| PureCoreData {
            author_id,
            source_tweet_id,
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .tweet_features(1, TweetFeatures::default())
                .fail_key(Source::TesPureCoreData, 1)
                .pure_core(3, core(30, None))
                .pure_core(4, core(40, Some(6)))
                .pure_core(5, core(50, Some(6)))
                .pure_core(6, core(60, None))
                .pure_core(7, core(70, Some(8)))
                .authors(&[30, 40, 50, 70])
                .user(
                    60,
                    GizmoduckUserResult {
                        user: Some(GizmoduckUser {
                            safety: Safety {
                                suspended: true,
                                ..Default::default()
                            },
                            ..Default::default()
                        }),
                        response_state: Some(UserResponseState::Found),
                    },
                ),
        );
        let wire = Wire::new(Arc::<InMemorySources>::clone(&sources));
        for (level, code) in [
            (0, tonic::Code::Unimplemented),
            (4, tonic::Code::Unimplemented),
            (
                ThriftLevel::QUOTED_TWEET_RULES.0,
                tonic::Code::Unimplemented,
            ),
            (
                ThriftLevel::IMMERSIVE_EXPANDED_RECOMMENDATIONS.0,
                tonic::Code::Unimplemented,
            ),
            (9999, tonic::Code::InvalidArgument),
        ] {
            let error = wire
                .send(vf_pb::EvaluateTweetsRequest {
                    safety_level: level,
                    ..Default::default()
                })
                .await
                .unwrap_err();
            assert_eq!(error.code(), code);
        }
        let tweets = vec![
            tweet_data(1, None),
            tweet_data(1, Some(2)),
            tweet_data(3, None),
            tweet_data(3, None),
        ];
        for (level, quoted, evaluated) in [
            (
                ThriftLevel::FILTER_ALL.0,
                Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                filtered(ThriftFilteredReason::SafetyResult(SafetyResult::new(
                    None,
                    Action::Drop(action::Drop::new(Some(DropReason::Unspecified(true)), None)),
                ))),
            ),
            (
                ThriftLevel::TIMELINE_HOME_HYDRATION.0,
                Outcome::Failed(vf_pb::Failed {}),
                encoded(&TweetFieldsResultState::Found(TweetFieldsResultFound::new(
                    None,
                ))),
            ),
        ] {
            let sent = wire
                .send(vf_pb::EvaluateTweetsRequest {
                    safety_level: level,
                    tweets: tweets.clone(),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert_eq!(
                sent,
                vec![
                    Outcome::Failed(vf_pb::Failed {}),
                    quoted,
                    evaluated.clone(),
                    evaluated,
                ]
            );
        }
        let suspended = filtered(ThriftFilteredReason::AuthorIsSuspended(true));
        for (tweet_ids, core_data_calls, outcomes) in [
            (vec![4, 6], 1, vec![suspended.clone(), suspended.clone()]),
            (vec![5, 7], 2, vec![suspended, not_found()]),
        ] {
            let calls_before = sources.keys(Source::TesPureCoreData).len();
            let sent = wire
                .send(vf_pb::EvaluateTweetsRequest {
                    safety_level: 8,
                    tweets: tweet_ids
                        .into_iter()
                        .map(|id| tweet_data(id, None))
                        .collect(),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert_eq!(
                sources.keys(Source::TesPureCoreData).len() - calls_before,
                core_data_calls
            );
            assert_eq!(sent, outcomes);
        }
    }

    #[tokio::test]
    async fn an_author_removed_community_post_and_its_retweet_drop_or_fail_with_the_lookup() {
        use crate::models::CommunityModeration;
        use std::num::NonZeroU64;
        let dropped = filtered(ThriftFilteredReason::SafetyResult(SafetyResult::new(
            None,
            Action::Drop(action::Drop::new(None, None)),
        )));
        for (fault, outcome) in [(None, dropped), (Some(Fault::Fails), failed())] {
            let mut world = InMemorySources::default()
                .pure_core(
                    4,
                    PureCoreData {
                        author_id: 40,
                        source_tweet_id: Some(6),
                        ..Default::default()
                    },
                )
                .tweet(6, 60)
                .tweet_features(
                    6,
                    TweetFeatures {
                        community_id: NonZeroU64::new(500),
                        ..Default::default()
                    },
                )
                .community_moderation(
                    6,
                    CommunityModeration {
                        is_hidden: false,
                        is_author_removed: true,
                    },
                )
                .authors(&[40, 60]);
            if let Some(fault) = fault {
                world = world.fault(Source::CommunityModeration, fault);
            }
            let sent = Wire::new(Arc::new(world))
                .send(vf_pb::EvaluateTweetsRequest {
                    safety_level: ThriftLevel::TIMELINE_HOME_HYDRATION.0,
                    viewer_id: Some(50),
                    tweets: vec![tweet_data(4, None), tweet_data(6, None)],
                })
                .await
                .unwrap();
            assert_eq!(sent, [outcome.clone(), outcome], "{fault:?}");
        }
    }

    #[tokio::test]
    async fn a_tweet_shown_standalone_and_quoted_gets_one_outcome_per_entry_in_request_order() {
        use crate::models::SafetyLabelType;
        let bounced = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(SafetyLabelType::BOUNCE.0, vf_pb::SafetyLabel::default())]),
        };
        let wire = Wire::new(Arc::new(
            InMemorySources::default()
                .tweet(1, 10)
                .labels(1, bounced)
                .authors(&[10]),
        ));
        let dropped = filtered(ThriftFilteredReason::TweetIsBounced(true));
        let tombstoned = filtered(ThriftFilteredReason::SafetyResult(SafetyResult::new(
            None,
            Action::Tombstone(action::Tombstone::new(
                Some(action::TombstoneReason::BOUNCED),
                None,
            )),
        )));
        for (level, viewer_id, outcomes) in [
            (
                ThriftLevel::TIMELINE_HOME_HYDRATION,
                Some(10),
                vec![shown(), tombstoned, shown()],
            ),
            (
                ThriftLevel::TIMELINE_HOME_HYDRATION,
                Some(VIEWER),
                vec![dropped.clone(); 3],
            ),
            (ThriftLevel::TIMELINE_HOME_HYDRATION, None, vec![dropped; 3]),
            (
                ThriftLevel::TIMELINE_HOME,
                Some(10),
                vec![
                    shown(),
                    Outcome::NotEvaluated(vf_pb::NotEvaluated {}),
                    shown(),
                ],
            ),
        ] {
            let sent = wire
                .send(vf_pb::EvaluateTweetsRequest {
                    safety_level: level.0,
                    viewer_id,
                    tweets: vec![
                        tweet_data(1, None),
                        tweet_data(1, Some(2)),
                        tweet_data(1, None),
                    ],
                })
                .await
                .unwrap();
            assert_eq!(sent, outcomes, "{level:?} {viewer_id:?}");
        }
    }

    #[tokio::test]
    async fn each_entry_answers_as_it_does_alone_whatever_else_the_batch_requests() {
        use crate::hydration::sources::suspended;
        use crate::models::SafetyLabelType;
        let bounced = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(SafetyLabelType::BOUNCE.0, vf_pb::SafetyLabel::default())]),
        };
        let wire = Wire::new(Arc::new(
            InMemorySources::default()
                .retweet(1, 10, 5, Some(20))
                .tweet(2, 30)
                .tweet(3, 10)
                .tweet(4, 10)
                .fail_key(Source::TesTweetForVisibility, 4)
                .tweet(5, 20)
                .labels(5, bounced)
                .retweet(6, 10, 7, Some(30))
                .tweet(7, 30)
                .authors(&[10, 20])
                .user(30, suspended()),
        ));
        let evaluate = |tweets: Vec<vf_pb::TweetData>| {
            let wire = &wire;
            async move {
                wire.send(vf_pb::EvaluateTweetsRequest {
                    safety_level: ThriftLevel::TIMELINE_HOME_HYDRATION.0,
                    viewer_id: Some(VIEWER),
                    tweets,
                })
                .await
                .unwrap()
            }
        };
        let bounced = filtered(ThriftFilteredReason::TweetIsBounced(true));
        let suspended = filtered(ThriftFilteredReason::AuthorIsSuspended(true));
        let entries = [
            (tweet_data(3, None), shown()),
            (tweet_data(1, None), bounced.clone()),
            (tweet_data(5, Some(9)), bounced.clone()),
            (tweet_data(4, None), failed()),
            (tweet_data(8, None), not_found()),
            (tweet_data(5, None), bounced),
            (tweet_data(2, None), suspended.clone()),
            (tweet_data(6, None), suspended.clone()),
            (tweet_data(7, Some(9)), suspended.clone()),
            (tweet_data(3, None), shown()),
            (tweet_data(2, Some(9)), suspended),
        ];
        for (tweet, outcome) in &entries {
            assert_eq!(
                evaluate(vec![*tweet]).await,
                std::slice::from_ref(outcome),
                "{tweet:?}"
            );
        }
        let (tweets, outcomes): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
        assert_eq!(evaluate(tweets).await, outcomes);
    }

    const VIEWER: u64 = 50;

    async fn evaluate_tweets(
        world: InMemorySources,
        level: ThriftLevel,
        tweet_ids: &[u64],
    ) -> Vec<Outcome> {
        Wire::new(Arc::new(world))
            .send(vf_pb::EvaluateTweetsRequest {
                safety_level: level.0,
                viewer_id: Some(VIEWER),
                tweets: tweet_ids
                    .iter()
                    .map(|&tweet_id| tweet_data(tweet_id, None))
                    .collect(),
            })
            .await
            .unwrap()
    }

    fn shown() -> Outcome {
        encoded(&TweetFieldsResultState::Found(TweetFieldsResultFound::new(
            None,
        )))
    }

    fn failed() -> Outcome {
        Outcome::Failed(vf_pb::Failed {})
    }

    fn not_found() -> Outcome {
        encoded(&TweetFieldsResultState::NotFound(
            TweetFieldsResultNotFound::new(false, false, None),
        ))
    }

    #[tokio::test(start_paused = true)]
    async fn a_requested_post_tes_reports_missing_sends_its_deletion_state_when_tweet_state_answers(
    ) {
        use crate::hydration::fallback_cache::FallbackCaches;
        use crate::hydration::tweet_source::TweetState;
        use crate::hydration::BEST_EFFORT_HYDRATION_TIMEOUT;
        use crate::metrics::families::TWEET_STATE;
        use crate::quote::{QuotePassMode, QuotePassRunner};
        use crate::scenario::{raw, Scenario};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .retweet(8, 10, 9, Some(10))
                .tweet(11, 99)
                .authors(&[10])
                .tweet_state(2, TweetState::BounceDeleted)
                .tweet_state(3, TweetState::Deleted)
                .tweet_state(4, TweetState::NotFound)
                .tweet_state(5, TweetState::Found)
                .tweet_state(6, TweetState::BounceDeleted)
                .fail_key(Source::TesTweetState, 6)
                .tweet_state(9, TweetState::Deleted)
        };
        let deleted = |bounce_deleted| {
            encoded(&TweetFieldsResultState::NotFound(
                TweetFieldsResultNotFound::new(true, bounce_deleted, None),
            ))
        };
        let sources = Arc::new(world());
        let (metrics, recording) = Metrics::recording();
        let wire = Wire::serving(
            Arc::<InMemorySources>::clone(&sources),
            FallbackCaches::default(),
            None,
            QuotePassRunner::new(QuotePassMode::Off),
            metrics,
        );
        let sent = wire
            .send(vf_pb::EvaluateTweetsRequest {
                safety_level: ThriftLevel::TIMELINE_HOME_HYDRATION.0,
                viewer_id: Some(VIEWER),
                tweets: [1, 2, 3, 4, 5, 6, 7, 8, 11]
                    .map(|tweet_id| tweet_data(tweet_id, None))
                    .to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(
            sent,
            [
                shown(),
                deleted(true),
                deleted(false),
                not_found(),
                not_found(),
                not_found(),
                not_found(),
                not_found(),
                not_found(),
            ]
        );
        assert_eq!(
            sources.keys(Source::TesTweetState),
            [vec![2, 3, 4, 5, 6, 7]]
        );
        let mut states: Vec<(String, f64)> = recording
            .take(&TWEET_STATE.family)
            .into_iter()
            .map(|sample| (sample.label("state").unwrap().to_owned(), sample.value))
            .collect();
        states.sort_by(|a, b| a.0.cmp(&b.0));
        let expected = [
            ("bounce_deleted", 1.0),
            ("deleted", 1.0),
            ("failed", 1.0),
            ("found", 1.0),
            ("not_found", 2.0),
        ];
        assert_eq!(
            states,
            expected.map(|(state, keys)| (state.to_owned(), keys))
        );

        let filter_tweets = Scenario::new(world(), SafetyLevel::TimelineHomeHydration);
        let filtered = filter_tweets.sources();
        filter_tweets.hydrate(&[raw(2, None)]).await;
        assert!(filtered.keys(Source::TesTweetState).is_empty());

        let hung = world().fault(Source::TesTweetState, Fault::Hangs);
        let started = tokio::time::Instant::now();
        assert_eq!(
            evaluate_tweets(hung, ThriftLevel::TIMELINE_HOME_HYDRATION, &[1, 2]).await,
            [shown(), not_found()]
        );
        assert_eq!(started.elapsed(), BEST_EFFORT_HYDRATION_TIMEOUT);

        let budget = BEST_EFFORT_HYDRATION_TIMEOUT / 2;
        let hung = Scenario::new(
            world().fault(Source::TesTweetState, Fault::Hangs),
            SafetyLevel::TimelineHomeHydration,
        )
        .retweet_merge(RetweetMerge::Merged);
        let started = tokio::time::Instant::now();
        xai_x_rpc::CallContext {
            deadline: Some(started + budget),
        }
        .scope(hung.hydrate(&[raw(2, None)]))
        .await;
        assert_eq!(started.elapsed(), budget);
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_flock_call_fails_a_post_only_through_a_must_have_node_it_carries() {
        use crate::hydration::{
            Hydrator, BEST_EFFORT_HYDRATION_TIMEOUT, HYDRATION_TIMEOUT, INBOUND_ALLOWANCE,
        };
        use crate::models::{RequestFacts, SafetyLabelType};
        use crate::rules::fixtures::{allow, dropped};
        use xai_visibility_filtering::models::FilteredReason;
        let protected = GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety: Safety {
                    is_protected: true,
                    ..Default::default()
                },
                ..Default::default()
            }),
            response_state: Some(UserResponseState::Found),
        };
        let spam = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(SafetyLabelType::SPAM.0, vf_pb::SafetyLabel::default())]),
        };
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .tweet(3, 30)
                .labels(3, spam.clone())
                .authors(&[10, 30])
                .user(20, protected.clone())
                .fault(Source::Flock, Fault::Hangs)
        };
        let verdicts = [
            allow(),
            dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
            dropped(FilteredReason::PossiblyUndesirable, "spam/drop/undesirable"),
        ];
        let served = vec![
            shown(),
            filtered(ThriftFilteredReason::AuthorIsProtected(true)),
            filtered(ThriftFilteredReason::SafetyResult(SafetyResult::new(
                None,
                Action::Drop(action::Drop::new(None, None)),
            ))),
        ];
        for (level, thrift, is_partial, outcomes, answered_at) in [
            (
                SafetyLevel::TimelineHomeHydration,
                ThriftLevel::TIMELINE_HOME_HYDRATION,
                false,
                served,
                BEST_EFFORT_HYDRATION_TIMEOUT,
            ),
            (
                SafetyLevel::TimelineHomeRecommendations,
                ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS,
                true,
                vec![failed(); 3],
                HYDRATION_TIMEOUT - INBOUND_ALLOWANCE,
            ),
            (
                SafetyLevel::TimelineHome,
                ThriftLevel::TIMELINE_HOME,
                true,
                vec![failed(); 3],
                HYDRATION_TIMEOUT - INBOUND_ALLOWANCE,
            ),
        ] {
            let post_evaluator = PostEvaluator::new(
                Arc::new(world()),
                RuleEngine::for_tests(),
                Arc::default(),
                Metrics::none(),
            );
            let request = |retweet_merge| PostBatch {
                facts: RequestFacts::logged_in(VIEWER),
                safety_level: level,
                candidates: [1, 2, 3]
                    .map(|tweet_id| {
                        PlacedCandidate::standalone(RawCandidate {
                            tweet_id: TweetId(tweet_id),
                            request_author_id: None,
                        })
                    })
                    .to_vec(),
                retweet_merge,
            };
            for (retweet_merge, endpoint) in [
                (RetweetMerge::Unmerged, Endpoint::FilterTweets),
                (RetweetMerge::Merged, Endpoint::EvaluateTweets),
            ] {
                let metrics = Metrics::none().request(endpoint, RequestLevel::Named(level));
                let (_, outcomes) = post_evaluator
                    .run(request(retweet_merge), &metrics, &mut NoTrace)
                    .await;
                assert!(
                    outcomes[1]
                        .evaluation
                        .fail_open_defaults()
                        .contains(Hydrator::Follows),
                    "{level:?}"
                );
                for (outcome, verdict) in outcomes.into_iter().zip(&verdicts) {
                    assert_eq!(
                        matches!(outcome.evaluation, Evaluation::Partial { .. }),
                        is_partial,
                        "{level:?} {:?}",
                        outcome.evaluation
                    );
                    assert_eq!(&outcome.evaluation.into_verdict(), verdict, "{level:?}");
                }
            }
            let started = tokio::time::Instant::now();
            assert_eq!(
                evaluate_tweets(world(), thrift, &[1, 2, 3]).await,
                outcomes,
                "{level:?}"
            );
            assert_eq!(started.elapsed(), answered_at, "{level:?}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_must_have_lookup_fails_exactly_the_posts_it_failed_for() {
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .tweet(3, 30)
                .authors(&[10, 20, 30])
        };
        for (world, outcomes) in [
            (
                world().fail_key(Source::SafetyLabels, 2),
                vec![shown(), failed(), shown()],
            ),
            (
                world().fault(Source::GizmoduckViewer, Fault::Fails),
                vec![failed(); 3],
            ),
        ] {
            assert_eq!(
                evaluate_tweets(world, ThriftLevel::TIMELINE_HOME_HYDRATION, &[1, 2, 3]).await,
                outcomes
            );
        }
    }

    #[test]
    fn a_conversation_control_limit_carries_its_prompt_in_the_request_language() {
        use crate::models::LimitedEngagementReason::ConversationControl;
        use crate::rules::fixtures::{blurred_and_limited, legacy_interstitial, limited};
        use xai_x_thrift::action::{
            AnyInterstitial, CtaLimitedActionPrompt, Interstitial, InterstitialReason,
            LimitedAction, LimitedActionCtaType, LimitedActionPrompt, LimitedActionsPolicy,
            LimitedEngagements, TweetInterstitial,
        };
        let limit = limited(ConversationControl, "rule");
        let composite = blurred_and_limited(legacy_interstitial("nsfw_rule"), limit.clone());
        for (language, subtext) in [
            ("", "Only some accounts can reply."),
            ("ja", "一部のアカウントのみが返信できます。"),
        ] {
            let context = TwitterContextViewer {
                request_language_code: language.to_string(),
                ..TwitterContextViewer::default()
            };
            let policies = ClientSwitches::for_tests().limited_actions_policies(
                Some(&context),
                None,
                None,
                [ConversationControl],
                &Metrics::none(),
            );
            let thrift_limit = LimitedEngagements::new(
                action::LimitedEngagementReason::ConversationControl(
                    action::ConversationControl::new(),
                ),
                LimitedActionsPolicy::new(vec![LimitedAction::new(
                    action::LimitedActionType::REPLY,
                    LimitedActionPrompt::CtaLimitedActionPrompt(CtaLimitedActionPrompt::new(
                        "Who can reply?".to_string(),
                        subtext.to_string(),
                        LimitedActionCtaType::SEE_CONVERSATION,
                    )),
                )]),
                "limited_replies".to_string(),
            );
            let found = |action| {
                encoded(&TweetFieldsResultState::Found(TweetFieldsResultFound::new(
                    ThriftFilteredReason::SafetyResult(SafetyResult::new(None, action)),
                )))
            };
            for (verdict, action) in [
                (&limit, Action::LimitedEngagements(thrift_limit.clone())),
                (
                    &composite,
                    Action::TweetInterstitial(TweetInterstitial {
                        interstitial: Some(AnyInterstitial::Interstitial(Interstitial::new(
                            InterstitialReason::ContainsNsfwMedia(true),
                            None,
                        ))),
                        limited_engagements: Some(thrift_limit.clone()),
                        ..TweetInterstitial::default()
                    }),
                ),
            ] {
                let evaluation = Evaluation::complete(verdict.clone());
                assert_eq!(
                    treatment::evaluate_tweets_outcome(
                        Some(&evaluation),
                        SafetyLevel::TimelineHomeHydration,
                        &policies
                    ),
                    found(action),
                    "{language:?} {verdict:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn the_forwarded_client_reaches_the_fetched_source_and_no_header_gets_the_defaults() {
        use crate::rules::fixtures::CLIENT_CLASSES;
        let world = Arc::new(
            InMemorySources::default()
                .pure_core(
                    4,
                    PureCoreData {
                        author_id: 40,
                        source_tweet_id: Some(6),
                        ..Default::default()
                    },
                )
                .tweet(6, 60)
                .community_post(6, 500),
        );
        let wire = Wire::new(Arc::<InMemorySources>::clone(&world));
        let class = CLIENT_CLASSES
            .iter()
            .find(|class| {
                class
                    .capability
                    .has(ClientSwitch::CommunityViewerRemovedLimits)
            })
            .unwrap();
        let context = TwitterContextViewer {
            client_application_id: class.app_id,
            user_agent: class.user_agent.into(),
            ..TwitterContextViewer::default()
        };
        let request = || vf_pb::EvaluateTweetsRequest {
            safety_level: ThriftLevel::TIMELINE_HOME_HYDRATION.0,
            viewer_id: Some(1),
            tweets: vec![tweet_data(4, None)],
        };
        wire.send_with_context(&context, request()).await.unwrap();
        assert_eq!(world.keys(Source::CommunityViewerRemoved), [vec![500]]);
        wire.send(request()).await.unwrap();
        assert_eq!(world.keys(Source::CommunityViewerRemoved), [vec![500]]);
    }
}
