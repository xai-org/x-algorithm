use super::{one_viewer_batch, source, Backends, CallContext, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::decode::author::{
    actor_batch, author_batch, AuthorColumn, DecodedActor, DecodedAuthor,
};
use crate::hydration::decode::viewer::{decode_viewer, DecodedViewer};
use crate::hydration::execute::fall_back;
use crate::hydration::hydrator::Edge;
use tracing::warn;
#[cfg(test)]
use {
    super::InMemoryCall,
    std::collections::HashMap,
    xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, UserResponseState},
};

impl SourceFetch for source::GizmoduckViewer {
    const BACKEND_LABEL: &'static str = "gizmoduck";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<DecodedViewer>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let viewer = backends
            .gizmoduck
            .get_viewer(viewer_id, input.fields)
            .await
            .inspect_err(|error| warn!(%error, "Gizmoduck viewer lookup failed; failing open"));
        let viewer = viewer.map(|user| Some(decode_viewer(user.as_ref(), input.fields)));
        Ok(one_viewer_batch(viewer_id, viewer))
    }

    #[cfg(test)]
    type World = HashMap<u64, GizmoduckUser>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed_with(input.viewer_id.as_slice(), |viewer_id| {
            Some(decode_viewer(world.get(&viewer_id), input.fields))
        })
    }
}

impl SourceFetch for source::GizmoduckAuthor {
    const BACKEND_LABEL: &'static str = "gizmoduck";
    const DECIDES_EXISTENCE: bool = true;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<DecodedAuthor>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let users = backends
            .gizmoduck
            .get_users(input.keys.to_vec(), input.fields)
            .await;
        Ok(author_batch(HydrationBatch::from_results(
            input.keys.iter().copied(),
            users,
        )))
    }

    fn fall_back(context: &CallContext<'_>, authors: Self::Reply) -> Self::Reply {
        let authors =
            fall_back::<AuthorColumn>(context.metrics.plain(), context.author_cache, authors);
        context.author_not_found.read(authors)
    }

    #[cfg(test)]
    type World = HashMap<u64, GizmoduckUserResult>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        author_batch(users(world, input, call))
    }
}

impl SourceFetch for source::GizmoduckActor {
    const BACKEND_LABEL: &'static str = "gizmoduck";
    const DECIDES_EXISTENCE: bool = true;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<DecodedActor>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let users = backends
            .gizmoduck
            .get_users(input.keys.to_vec(), input.fields)
            .await;
        Ok(actor_batch(HydrationBatch::from_results(
            input.keys.iter().copied(),
            users,
        )))
    }

    #[cfg(test)]
    type World = HashMap<u64, GizmoduckUserResult>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        actor_batch(users(world, input, call))
    }
}

#[cfg(test)]
fn users(
    world: &HashMap<u64, GizmoduckUserResult>,
    input: &Input<'_>,
    call: &InMemoryCall<'_>,
) -> RawHydrationBatch<GizmoduckUserResult> {
    let users = call.keyed(input.keys, world);
    if call.answers_partial() {
        users.map(|user| GizmoduckUserResult {
            response_state: Some(UserResponseState::Partial),
            ..user
        })
    } else {
        users
    }
}

#[cfg(test)]
mod tests {
    use crate::clients::socialgraph_client::Graph;
    use crate::hydration::decode::author::fallback_cache;
    use crate::hydration::execute::tests::{call_labels, executor, VIEWER};
    use crate::hydration::fallback_cache::FallbackCaches;
    use crate::hydration::sources::{suspended, Fault, InMemorySources, Source};
    use crate::hydration::HydrationRequest;
    use crate::hydration::{
        Cause, HydrationPlan, Hydrator, Hydrators, Lookup, Relation, Unresolved,
    };
    use crate::metrics::{Endpoint, Metrics, RequestLevel};
    use crate::models::{AuthorId, RequestFacts, RetweetMerge};
    use crate::rules::SafetyLevel;
    use crate::scenario::{protected_user, raw, reaction_to};
    use xai_core_entities::gizmoduck_client::QueryFields;

    #[tokio::test]
    async fn a_reacted_to_author_lookup_neither_reads_nor_writes_the_author_fallback_cache() {
        let sources = InMemorySources::default()
            .tweet(1, 70)
            .tweet(2, 10)
            .tweet_features(2, reaction_to(7, 70))
            .tweet(3, 10)
            .tweet_features(3, reaction_to(8, 71))
            .tweet(4, 71)
            .authors(&[10])
            .user(70, protected_user(70))
            .user(71, protected_user(71));
        let caches = FallbackCaches {
            tweets: None,
            authors: Some(fallback_cache(8)),
        };
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .retweet_merge(RetweetMerge::Merged)
            .fallback_caches(caches);
        let first = scenario.hydrate(&[raw(1, None), raw(3, None)]).await;
        assert!(first.candidates[1].post.reacted_to_author_is_protected);

        let sources = scenario.sources();
        sources.break_source(Source::GizmoduckAuthor, Fault::Fails);
        sources.break_source((Relation::ReactedTo, Source::GizmoduckAuthor), Fault::Fails);
        let second = scenario
            .hydrate(&[raw(1, None), raw(2, None), raw(4, None)])
            .await;
        assert!(second.candidates[0].author.features.is_protected);
        assert!(!second.candidates[1].post.reacted_to_author_is_protected);
        assert_eq!(
            second.failed_nodes,
            [
                Ok(Hydrators::empty()),
                Ok(Hydrators::of(Hydrator::ReactedToAuthorSafety)),
                Err(Unresolved {
                    lookup: Lookup::Author,
                    cause: Cause::Failed,
                }),
            ]
        );
    }

    #[tokio::test]
    async fn the_author_cache_serves_the_last_known_author_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended());
        let raw = [raw(1, None)];
        let caches = FallbackCaches {
            tweets: None,
            authors: Some(fallback_cache(8)),
        };
        let scenario = executor(sources, SafetyLevel::TimelineHome, None).fallback_caches(caches);
        let sources = scenario.sources();
        let first = scenario.hydrate(&raw).await;
        assert!(first.candidates[0].author.features.is_suspended);

        sources.break_source(Source::GizmoduckAuthor, Fault::Fails);
        let second = scenario.hydrate(&raw).await;
        assert!(second.candidates[0].author.features.is_suspended);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn user_rows_ask_the_actor_source_by_user_without_pure_core_and_ask_a_repeated_user_once()
    {
        use crate::hydration::{HydratedUser, RequestRows};
        let sources = InMemorySources::default()
            .user(10, suspended())
            .authors(&[20])
            .edge(Graph::Blocks, VIEWER, 20)
            .fail_key(Source::GizmoduckActor, 30);
        let plan = HydrationPlan::with_rows(
            SafetyLevel::QuotedTweetRules,
            RequestRows::Users,
            Hydrators::of(Hydrator::AuthorSafety).with(Hydrator::Blocks),
        );
        let caches = FallbackCaches {
            tweets: None,
            authors: None,
        };
        let users = [10, 20, 10, 30].map(AuthorId).to_vec();
        let hydration = plan
            .hydrate(
                &sources,
                HydrationRequest::for_users(
                    RequestFacts::logged_in(VIEWER),
                    users,
                    &caches,
                    &Metrics::none().request(
                        Endpoint::EvaluateTweets,
                        RequestLevel::Named(SafetyLevel::QuotedTweetRules),
                    ),
                ),
            )
            .await;
        assert_eq!(call_labels(&sources), ["Flock blocks", "GizmoduckActor"]);
        assert_eq!(sources.keys(Source::GizmoduckActor), [vec![10, 20, 30]]);
        assert!(matches!(
            hydration.user(AuthorId(10)),
            Some(HydratedUser::Resolved { user, edges, failed })
                if user.features.is_suspended && !edges.contains(Hydrator::Blocks) && failed.is_empty()
        ));
        assert!(matches!(
            hydration.user(AuthorId(20)),
            Some(HydratedUser::Resolved { edges, failed, .. })
                if edges.contains(Hydrator::Blocks) && failed.is_empty()
        ));
        assert!(matches!(
            hydration.user(AuthorId(30)),
            Some(HydratedUser::Unresolved(Cause::Failed))
        ));
    }

    #[tokio::test]
    async fn gizmoduck_calls_ask_for_every_field_any_level_reads() {
        use QueryFields::{ACCOUNT, EXTENDED_PROFILE, LABELS, SAFETY};
        for (level, viewer_fields) in [
            (
                SafetyLevel::TimelineHome,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY],
            ),
            (
                SafetyLevel::TimelineHomeHydration,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY, LABELS],
            ),
        ] {
            let sources = InMemorySources::default().tweet(1, 10);
            let scenario = executor(sources, level, Some(VIEWER));
            let sources = scenario.sources();
            scenario.hydrate(&[raw(1, None)]).await;
            assert_eq!(
                sources.fields(Source::GizmoduckViewer),
                [viewer_fields],
                "{level:?}"
            );
            assert_eq!(
                sources.fields(Source::GizmoduckAuthor),
                [vec![SAFETY, LABELS]],
                "{level:?}"
            );
        }
    }
}
