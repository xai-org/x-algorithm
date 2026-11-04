use crate::hydration::sources::Source;
use crate::hydration::store::KeyOrigin;
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{AuthorId, PureCore, RetweetMerge, TweetFeatures, TweetId};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Relation {
    RetweetSource,
    ConversationRoot,
    ReactedTo,
    QuoteOuter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Related {
    pub(super) tweet_id: TweetId,
    pub(super) author: Option<AuthorId>,
}

pub(super) struct RelationSpec {
    pub(super) trigger: Trigger,
    pub(super) role: Role,
    pub(super) fetch: Fetch,
}

#[derive(Clone, Copy)]
pub(super) enum Trigger {
    PureCore(fn(&PureCore) -> Option<Related>),
    Tweet(fn(&TweetFeatures) -> Option<Related>),
    Key(KeyOrigin),
    QuoteEntry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    Judged,
    Read,
}

#[derive(Clone, Copy)]
pub(super) enum Fetch {
    Own(&'static [Source]),
    Shared,
}

impl Relation {
    pub(super) const fn spec(self) -> RelationSpec {
        match self {
            Relation::RetweetSource => RelationSpec {
                trigger: Trigger::PureCore(retweet_source),
                role: Role::Judged,
                fetch: Fetch::Shared,
            },
            Relation::ConversationRoot => RelationSpec {
                trigger: Trigger::Key(KeyOrigin::ConversationRootToFetch),
                role: Role::Read,
                fetch: Fetch::Own(&[Source::TesPureCoreData]),
            },
            Relation::ReactedTo => RelationSpec {
                trigger: Trigger::Tweet(reacted_to),
                role: Role::Read,
                fetch: Fetch::Own(&[Source::SafetyLabels, Source::GizmoduckAuthor]),
            },
            Relation::QuoteOuter => RelationSpec {
                trigger: Trigger::QuoteEntry,
                role: Role::Read,
                fetch: Fetch::Own(&[Source::Flock]),
            },
        }
    }
}

impl Relation {
    pub(super) const fn owns(self, source: Source) -> bool {
        let Fetch::Own(mut rest) = self.spec().fetch else {
            return false;
        };
        while let [owned, tail @ ..] = rest {
            if *owned as u8 == source as u8 {
                return true;
            }
            rest = tail;
        }
        false
    }

    pub(super) fn is_on(self, retweet_merge: RetweetMerge, planned: Hydrators) -> bool {
        match self.spec().role {
            Role::Judged => retweet_merge == RetweetMerge::Merged,
            Role::Read => planned
                .iter()
                .any(|node| node.spec().relation == Some(self)),
        }
    }
}

impl Trigger {
    pub(super) const fn input(self) -> Option<Hydrator> {
        match self {
            Trigger::PureCore(_) | Trigger::QuoteEntry => Some(Hydrator::PureCore),
            Trigger::Tweet(_) => Some(Hydrator::Tweet),
            Trigger::Key(origin) => origin.input(),
        }
    }

    pub(super) const fn reads(self) -> Hydrators {
        match self {
            Trigger::PureCore(_) | Trigger::QuoteEntry => Hydrators::of(Hydrator::PureCore),
            Trigger::Tweet(_) => Hydrators::of(Hydrator::Tweet),
            Trigger::Key(origin) => origin.reads(),
        }
    }

    pub(super) const fn needs_viewer(self) -> bool {
        match self {
            Trigger::PureCore(_) | Trigger::Tweet(_) | Trigger::QuoteEntry => false,
            Trigger::Key(origin) => origin.needs_viewer(),
        }
    }
}

fn retweet_source(core: &PureCore) -> Option<Related> {
    Some(Related {
        tweet_id: core.source_tweet_id?,
        author: core.source_author_id,
    })
}

pub(super) fn quoting_author(core: &PureCore) -> AuthorId {
    core.source_author_id.unwrap_or(core.author_id)
}

fn reacted_to(tweet: &TweetFeatures) -> Option<Related> {
    let context = tweet.reaction_context?;
    Some(Related {
        tweet_id: TweetId(context.reacted_to_post_id),
        author: Some(AuthorId(context.reacted_to_user_id)),
    })
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let spec = self.spec();
        let reads: Vec<&str> = spec.trigger.reads().iter().map(<&str>::from).collect();
        write!(
            f,
            "expand {} after: {}",
            <&str>::from(self),
            reads.join(",")
        )?;
        match spec.role {
            Role::Judged => f.write_str(" role: judged (retweet merge)")?,
            Role::Read => f.write_str(" role: read")?,
        }
        if spec.trigger.needs_viewer() {
            f.write_str(" (skipped logged out)")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::clients::socialgraph_client::{EdgeQuery, Graph};
    use crate::hydration::execute::tests::{executor, VIEWER};
    use crate::hydration::fetcher::FetcherId;
    use crate::hydration::sources::{control, Fault, InMemorySources, Source};
    use crate::hydration::{
        Hydrator, Hydrators, Lookup, Relation, Unresolved, BEST_EFFORT_HYDRATION_TIMEOUT,
        HYDRATION_TIMEOUT,
    };
    use crate::metrics::families::HYDRATOR_TWEET_IDS;
    use crate::metrics::Metrics;
    use crate::models::{Evaluation, Placement, RetweetMerge, SafetyLabelType};
    use crate::rules::{RuleEngine, SafetyLevel};
    use crate::scenario::{label_map, protected_user, raw, reaction_to};
    use std::collections::BTreeSet;
    use xai_core_entities::entities::{ConversationControlArm, PureCoreData};

    #[tokio::test(start_paused = true)]
    async fn only_the_blocked_by_node_a_post_reads_can_become_its_fail_open_default() {
        use ConversationControlArm::Subscribers;
        use Hydrator::BlockedByConversationRoot;
        let level = SafetyLevel::TimelineHomeHydration;
        let reply = |conversation_id, parent_id, parent_author_id| PureCoreData {
            author_id: 10,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(parent_id),
            in_reply_to_user_id: Some(parent_author_id),
            ..Default::default()
        };
        let control_fails =
            || InMemorySources::default().fault(Source::TesConversationControl, Fault::Fails);
        let rows = [
            (
                "a conversation root post whose conversation control call fails",
                control_fails().tweet(1, 10),
                &[1][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a conversation root post whose conversation root select claims its author and times out",
                InMemorySources::default()
                    .tweet(1, 10)
                    .control(1, control(Subscribers, 10, &[]))
                    .fault(
                        Source::TesPureCoreData,
                        Fault::Delays(BEST_EFFORT_HYDRATION_TIMEOUT / 2),
                    )
                    .hang_graph(Graph::SuperFollows),
                &[1][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a direct reply whose conversation control call fails",
                control_fails().pure_core(1, reply(100, 100, 30)),
                &[1][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a nested reply whose conversation root author is in the batch and whose conversation control call fails",
                control_fails().pure_core(1, reply(100, 2, 11)).pure_core(
                    2,
                    PureCoreData {
                        author_id: 11,
                        ..reply(100, 100, 30)
                    },
                ),
                &[1, 2][..],
                BlockedByConversationRoot,
                false,
            ),
            (
                "a nested reply whose conversation root is outside the batch and whose conversation control call fails",
                control_fails().pure_core(1, reply(100, 200, 11)),
                &[1][..],
                BlockedByConversationRoot,
                true,
            ),
        ];
        let engine = RuleEngine::for_tests();
        for (name, sources, batch, node, reports) in rows {
            let hydrated = executor(sources.authors(&[10, 11, 30]), level, Some(VIEWER))
                .hydrate(&batch.iter().map(|&id| raw(id, None)).collect::<Vec<_>>())
                .await;
            let evaluation = engine.evaluate(
                level,
                &hydrated.viewer_features,
                &hydrated.candidates[0],
                Placement::Standalone,
            );
            assert_eq!(
                evaluation.fail_open_defaults().contains(node),
                reports,
                "{name}: {node:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_conversation_root_post_whose_author_blocks_the_viewer_reads_only_the_author_block() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .control(1, control(ConversationControlArm::Community, 10, &[]))
            .edge(Graph::Blocks, 10, VIEWER);
        let hydrated = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .hydrate(&[raw(1, None)])
            .await;
        let candidate = &hydrated.candidates[0];
        assert!(candidate.edges.contains(Hydrator::BlockedByAuthor));
        assert!(!candidate.post.conversation_root_author_blocks_viewer);
    }

    #[tokio::test]
    async fn a_reply_asks_whether_its_conversation_roots_author_blocks_the_viewer() {
        let reply = |author_id, conversation_id, parent_id, parent_author_id| PureCoreData {
            author_id,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(parent_id),
            in_reply_to_user_id: Some(parent_author_id),
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .pure_core(1, reply(10, 100, 100, 30))
            .pure_core(2, reply(11, 100, 1, 10))
            .pure_core(3, reply(12, 200, 201, 98))
            .tweet(200, 40)
            .authors(&[10, 11, 12]);
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        scenario
            .hydrate(&[raw(1, None), raw(2, None), raw(3, None)])
            .await;
        let sources = scenario.sources();
        assert_eq!(
            sources.selects(),
            [
                vec![
                    EdgeQuery::forward(Graph::Follows, vec![10, 11, 12]),
                    EdgeQuery::reverse(Graph::Blocks, vec![10, 11, 12, 30]),
                ],
                vec![EdgeQuery::reverse(Graph::Blocks, vec![40])],
            ]
        );
        assert_eq!(sources.keys(Source::TesPureCoreData), [vec![1, 2, 3]]);
        assert_eq!(
            sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
            [vec![200]]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_conversation_root_post_fails_only_the_nested_replies_under_it() {
        use Hydrator::{BlockedByConversationRootPost, ConversationRootPost};
        let nested = |author_id, conversation_id| PureCoreData {
            author_id,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(conversation_id + 1),
            in_reply_to_user_id: Some(99),
            ..Default::default()
        };
        let replies = || {
            InMemorySources::default()
                .pure_core(1, nested(10, 100))
                .tweet(2, 20)
                .pure_core(3, nested(11, 300))
                .pure_core(4, nested(12, 100))
                .authors(&[10, 11, 12, 20])
                .edge(Graph::Blocks, 30, VIEWER)
        };
        let world = || replies().tweet(100, 30).tweet(300, 31);
        let author_select = vec![
            EdgeQuery::forward(Graph::Follows, vec![10, 11, 12, 20]),
            EdgeQuery::reverse(Graph::Blocks, vec![10, 11, 12, 20]),
        ];
        let root_failed: Result<Hydrators, Unresolved> =
            Ok(Hydrators::of(ConversationRootPost).with(BlockedByConversationRootPost));
        let complete: Result<Hydrators, Unresolved> = Ok(Hydrators::empty());
        let rows = [
            (
                "root call times out",
                world().key_latency(
                    (Relation::ConversationRoot, Source::TesPureCoreData),
                    100,
                    BEST_EFFORT_HYDRATION_TIMEOUT * 2,
                ),
                [root_failed, complete, root_failed, root_failed],
                vec![author_select.clone()],
                vec![("conversation_root_pure_core_data", "timeout", 3)],
            ),
            (
                "one root key fails",
                world().fail_key((Relation::ConversationRoot, Source::TesPureCoreData), 100),
                [root_failed, complete, complete, root_failed],
                vec![
                    author_select.clone(),
                    vec![EdgeQuery::reverse(Graph::Blocks, vec![31])],
                ],
                vec![
                    ("conversation_root_author_blocked_by", "success", 1),
                    ("conversation_root_pure_core_data", "error", 2),
                    ("conversation_root_pure_core_data", "success", 1),
                ],
            ),
            (
                "no root is found",
                replies(),
                [complete; 4],
                vec![author_select],
                vec![("conversation_root_pure_core_data", "success", 3)],
            ),
        ];
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let level = SafetyLevel::TimelineHomeHydration;
        for (name, sources, failed, selects, tweet_ids) in rows {
            let (metrics, recording) = Metrics::recording();
            let scenario = executor(sources, level, Some(VIEWER)).metrics(metrics);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            let mut recorded: Vec<(String, String, usize)> = recording
                .take(&HYDRATOR_TWEET_IDS.family)
                .iter()
                .filter(|sample| {
                    matches!(
                        sample.label("call"),
                        Some(
                            "conversation_root_pure_core_data"
                                | "conversation_root_author_blocked_by"
                        )
                    )
                })
                .map(|sample| {
                    let label = |name| sample.label(name).unwrap_or_default().to_owned();
                    (label("call"), label("result"), sample.value as usize)
                })
                .collect();
            recorded.sort();
            let tweet_ids: Vec<(String, String, usize)> = tweet_ids
                .into_iter()
                .map(|(call, result, count)| (call.to_owned(), result.to_owned(), count))
                .collect();
            assert_eq!(recorded, tweet_ids, "{name}");
            assert_eq!(hydrated.failed_nodes, failed, "{name}");
            assert!(
                hydrated
                    .candidates
                    .iter()
                    .all(|c| !c.post.conversation_root_author_blocks_viewer),
                "{name}"
            );
            assert_eq!(
                sources.keys(Source::TesPureCoreData),
                [vec![1, 2, 3, 4]],
                "{name}"
            );
            assert_eq!(
                sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
                [vec![100, 300]],
                "{name}"
            );
            assert_eq!(sources.selects(), selects, "{name}");
        }
    }

    #[tokio::test]
    async fn a_reaction_post_asks_its_own_calls_for_its_reacted_to_posts_labels_and_authors_safety()
    {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet_features(1, reaction_to(7, 70))
            .tweet(2, 20)
            .tweet_features(2, reaction_to(1, 20))
            .tweet(3, 30)
            .labels(7, label_map(SafetyLabelType::NSFW_HIGH_PRECISION))
            .authors(&[10, 20, 30])
            .user(70, protected_user(70));
        let requested = [raw(1, None), raw(2, None), raw(3, None)];
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let sources = scenario.sources();
        let hydrated = scenario.hydrate(&requested).await;
        for (fetcher, asked) in [
            (Source::SafetyLabels.into(), vec![1, 2, 3]),
            (
                FetcherId::from((Relation::ReactedTo, Source::SafetyLabels)),
                vec![1, 7],
            ),
            (Source::GizmoduckAuthor.into(), vec![10, 20, 30]),
            (
                (Relation::ReactedTo, Source::GizmoduckAuthor).into(),
                vec![20, 70],
            ),
        ] {
            let mut keys: Vec<u64> = sources.keys(fetcher).concat();
            keys.sort_unstable();
            assert_eq!(keys, asked, "{fetcher:?}");
        }
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.post
                        .reacted_to_post_safety_labels
                        .has_label(SafetyLabelType::NSFW_HIGH_PRECISION),
                    c.post.reacted_to_author_is_protected,
                ))
                .collect::<Vec<_>>(),
            [(true, true), (false, false), (false, false)]
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_batch_without_a_reaction_post_sends_no_reacted_to_call() {
        let plain = executor(
            InMemorySources::default().tweet(3, 30).authors(&[30]),
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
        );
        plain.hydrate(&[raw(3, None)]).await;
        assert_eq!(plain.sources().keys(Source::SafetyLabels), [vec![3]]);
        assert_eq!(plain.sources().keys(Source::GizmoduckAuthor), [vec![30]]);
        for source in [
            (Relation::ReactedTo, Source::SafetyLabels),
            (Relation::ReactedTo, Source::GizmoduckAuthor),
        ] {
            assert!(plain.sources().keys(source).is_empty(), "{source:?}");
        }
    }

    #[tokio::test]
    async fn a_failed_or_missing_reacted_to_lookup_shows_its_reaction_post_unrestricted() {
        use crate::rules::fixtures::allow;
        use Hydrator::{ReactedToAuthorSafety, ReactedToPostSafetyLabels};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, reaction_to(7, 70))
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let shown = |fail_open_defaults| Evaluation::Complete {
            verdict: allow(),
            fail_open_defaults,
        };
        let rows = [
            (
                "reacted-to post's labels fail",
                world()
                    .labels(7, label_map(SafetyLabelType::NSFW_HIGH_PRECISION))
                    .fail_key((Relation::ReactedTo, Source::SafetyLabels), 7),
                Hydrators::of(ReactedToPostSafetyLabels),
            ),
            (
                "reacted-to author fails",
                world()
                    .user(70, protected_user(70))
                    .fail_key((Relation::ReactedTo, Source::GizmoduckAuthor), 70),
                Hydrators::of(ReactedToAuthorSafety),
            ),
            (
                "reacted-to post and author not found",
                world(),
                Hydrators::empty(),
            ),
        ];
        let raw = [raw(1, None), raw(2, None)];
        for (name, sources, failed) in rows {
            let evaluations = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
                .retweet_merge(RetweetMerge::Merged)
                .evaluate(&raw)
                .await
                .evaluations();
            assert_eq!(
                evaluations,
                [shown(failed), shown(Hydrators::empty())],
                "{name}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_partial_slow_or_hung_reacted_to_lookup_leaves_the_requests_lookups_of_its_keys_complete(
    ) {
        use crate::rules::fixtures::allow;
        use Hydrator::{ReactedToAuthorSafety, ReactedToPostSafetyLabels};
        fn world() -> InMemorySources {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, reaction_to(7, 20))
                .key_latency(
                    Source::TesPureCoreData,
                    1,
                    BEST_EFFORT_HYDRATION_TIMEOUT / 10,
                )
                .authors(&[10, 20, 30])
        }
        type World = fn() -> InMemorySources;
        type Broken = fn(InMemorySources) -> InMemorySources;
        let worlds: [(&str, World); 3] = [
            ("reacted-to author also writes a requested post", || {
                world().tweet(2, 20)
            }),
            (
                "reacted-to author also writes a retweet's source the share names no author for",
                || world().retweet(2, 30, 5, None).tweet(5, 20),
            ),
            ("reacted-to post is also a retweet's source", || {
                world().retweet(2, 30, 7, Some(20)).tweet(7, 20)
            }),
        ];
        let both = Hydrators::of(ReactedToPostSafetyLabels).with(ReactedToAuthorSafety);
        let faults: [(&str, Broken, Hydrators); 4] = [
            (
                "fails",
                |sources| {
                    sources
                        .fail_key((Relation::ReactedTo, Source::SafetyLabels), 7)
                        .fail_key((Relation::ReactedTo, Source::GizmoduckAuthor), 20)
                },
                both,
            ),
            (
                "outlasts its ceiling",
                |sources| {
                    let slow = BEST_EFFORT_HYDRATION_TIMEOUT * 2;
                    sources
                        .key_latency((Relation::ReactedTo, Source::SafetyLabels), 7, slow)
                        .key_latency((Relation::ReactedTo, Source::GizmoduckAuthor), 20, slow)
                },
                both,
            ),
            (
                "hangs",
                |sources| {
                    sources
                        .fault((Relation::ReactedTo, Source::SafetyLabels), Fault::Hangs)
                        .fault((Relation::ReactedTo, Source::GizmoduckAuthor), Fault::Hangs)
                },
                both,
            ),
            (
                "answers partial",
                |sources| {
                    sources.fault(
                        (Relation::ReactedTo, Source::GizmoduckAuthor),
                        Fault::AnswersPartial,
                    )
                },
                Hydrators::of(ReactedToAuthorSafety),
            ),
        ];
        let requested = [raw(1, None), raw(2, None)];
        for (world, sources) in worlds {
            for (fault, broken, fail_open_defaults) in faults {
                let reaction = Evaluation::Complete {
                    verdict: allow(),
                    fail_open_defaults,
                };
                let scenario = executor(
                    broken(sources()),
                    SafetyLevel::TimelineHomeHydration,
                    Some(VIEWER),
                )
                .retweet_merge(RetweetMerge::Merged);
                let evaluations = scenario.evaluate(&requested).await.evaluations();
                assert_eq!(
                    evaluations,
                    [reaction, Evaluation::complete(allow())],
                    "{world}, the reacted-to lookup {fault}"
                );
                let hydrated = scenario.hydrate(&requested).await;
                assert_eq!(
                    hydrated.failed_nodes,
                    [Ok(fail_open_defaults), Ok(Hydrators::empty())],
                    "{world}, the reacted-to lookup {fault}"
                );
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_conversation_root_that_is_also_a_retweet_source_is_fetched_by_both_calls() {
        use crate::rules::fixtures::allow;
        let world_naming = |source_author| {
            InMemorySources::default()
                .pure_core(
                    1,
                    PureCoreData {
                        author_id: 10,
                        conversation_id: Some(100),
                        in_reply_to_tweet_id: Some(101),
                        in_reply_to_user_id: Some(99),
                        ..Default::default()
                    },
                )
                .retweet(2, 20, 100, source_author)
                .tweet(100, 30)
                .authors(&[10, 20, 30])
        };
        let world = || world_naming(Some(30));
        let root_failed = Evaluation::Complete {
            verdict: allow(),
            fail_open_defaults: Hydrators::of(Hydrator::BlockedByConversationRootPost),
        };
        let complete = Evaluation::complete(allow());
        let slow = BEST_EFFORT_HYDRATION_TIMEOUT * 2;
        let rows = [
            ("both answer", world(), [complete.clone(), complete.clone()]),
            (
                "the root call fails",
                world().fail_key((Relation::ConversationRoot, Source::TesPureCoreData), 100),
                [root_failed.clone(), complete.clone()],
            ),
            (
                "the root call times out",
                world().key_latency(
                    (Relation::ConversationRoot, Source::TesPureCoreData),
                    100,
                    slow,
                ),
                [root_failed.clone(), complete.clone()],
            ),
            (
                "the retweet-source call fails",
                world().fail_key(Source::TesPureCoreData, 100),
                [complete.clone(), Evaluation::Failed(Lookup::SharedTweet)],
            ),
            (
                "the root's blocked-by select claims an unauthored source's author and times out",
                world_naming(None)
                    .key_latency(
                        Source::TesPureCoreData,
                        100,
                        BEST_EFFORT_HYDRATION_TIMEOUT / 2,
                    )
                    .key_latency(Source::Flock, 30, HYDRATION_TIMEOUT / 2),
                [
                    root_failed.clone(),
                    Evaluation::Complete {
                        verdict: allow(),
                        fail_open_defaults: Hydrators::of(Hydrator::BlockedByAuthor),
                    },
                ],
            ),
        ];
        for (name, sources, expected) in rows {
            let (metrics, recording) = Metrics::recording();
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
                .retweet_merge(RetweetMerge::Merged)
                .metrics(metrics);
            let evaluations = scenario
                .evaluate(&[raw(1, None), raw(2, None)])
                .await
                .evaluations();
            let calls: BTreeSet<String> = recording
                .take(&HYDRATOR_TWEET_IDS.family)
                .iter()
                .filter(|sample| sample.label("backend") == Some("tes"))
                .filter_map(|sample| Some(sample.label("call")?.to_owned()))
                .filter(|call| call.ends_with("pure_core_data"))
                .collect();
            assert_eq!(
                calls,
                BTreeSet::from([
                    "conversation_root_pure_core_data".to_owned(),
                    "tweet_pure_core_data".to_owned()
                ]),
                "{name}"
            );
            let sources = scenario.sources();
            assert_eq!(
                sources.keys(Source::TesPureCoreData),
                [vec![1, 2], vec![100]],
                "{name}"
            );
            assert_eq!(
                sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
                [vec![100]],
                "{name}"
            );
            assert_eq!(evaluations, expected, "{name}");
        }
    }
}
