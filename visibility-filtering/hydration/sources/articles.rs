use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::decode::article::decode_lifecycle;
use crate::hydration::hydrator::Edge;
use crate::models::ArticleLifecycle;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashMap};

impl SourceFetch for source::ArticleLifecycle {
    const BACKEND_LABEL: &'static str = "articles";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<ArticleLifecycle>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let rows = backends.articles.lifecycles(input.keys).await;
        Ok(lifecycle_batch(HydrationBatch::from_results(
            input.keys.iter().copied(),
            rows,
        )))
    }

    #[cfg(test)]
    type World = HashMap<u64, ArticleLifecycle>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

impl SourceFetch for source::ArticleMediaEntities {
    const BACKEND_LABEL: &'static str = "articles";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let rows = backends
            .articles
            .media_entities(input.viewer_id, input.keys)
            .await;
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            rows,
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, bool>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

impl SourceFetch for source::ArticleCoverMedia {
    const BACKEND_LABEL: &'static str = "articles";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let rows = backends
            .articles
            .cover_media(input.viewer_id, input.keys)
            .await;
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            rows,
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, bool>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

fn lifecycle_batch(rows: RawHydrationBatch<i32>) -> RawHydrationBatch<ArticleLifecycle> {
    let decoded = rows.into_hydrated().into_iter().map(|(id, row)| {
        let lifecycle = match row {
            Hydrated::Found(value) => {
                decode_lifecycle(value).map_or(Hydrated::Failed, Hydrated::Found)
            }
            Hydrated::Partial(value) => {
                decode_lifecycle(value).map_or(Hydrated::Failed, Hydrated::Partial)
            }
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed => Hydrated::Failed,
            Hydrated::Unanswered => Hydrated::Unanswered,
        };
        (id, lifecycle)
    });
    HydrationBatch::from_hydrated(decoded.collect())
}

#[cfg(test)]
mod tests {
    use crate::hydration::execute::tests::{article_tweet, executor, VIEWER};
    use crate::hydration::sources::{InMemorySources, Source};
    use crate::hydration::Hydrators;
    use crate::models::{ArticleLifecycle, Evaluation};
    use crate::rules::SafetyLevel;
    use crate::scenario::raw;

    #[tokio::test]
    async fn each_article_column_asks_each_distinct_article_once() {
        use ArticleLifecycle::{Draft, Published};
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        for viewer_id in [Some(VIEWER), None] {
            let sources = InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .authors(&[10])
                .tweet_features(1, article_tweet(70))
                .tweet_features(2, article_tweet(70))
                .tweet_features(3, article_tweet(80))
                .lifecycle(70, Draft)
                .lifecycle(80, Published);
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            for source in [
                Source::ArticleLifecycle,
                Source::ArticleMediaEntities,
                Source::ArticleCoverMedia,
            ] {
                assert_eq!(
                    sources.keys(source),
                    [vec![70, 80]],
                    "{viewer_id:?} {source:?}"
                );
            }
            for source in [Source::ArticleMediaEntities, Source::ArticleCoverMedia] {
                assert_eq!(sources.viewers(source), [viewer_id], "{source:?}");
            }
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| c.post.article_lifecycle)
                    .collect::<Vec<_>>(),
                [Some(Draft), Some(Draft), Some(Published), None],
                "{viewer_id:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_failed_article_column_reads_as_no_media_and_leaves_the_post_complete() {
        use crate::hydration::sources::found_user;
        use crate::models::NsfwViewerDropReason;
        use crate::rules::fixtures::{allow, nsfw_viewer_dropped};
        use xai_core_entities::entities::{ExtendedProfile, GizmoduckUser};
        let mut author = found_user(10);
        author.user.as_mut().unwrap().safety.nsfw_user = true;
        let dropped = nsfw_viewer_dropped(
            NsfwViewerDropReason::IsUnderage,
            "sensitive_viewer_underage/drop",
        );
        for (failing, verdict) in [
            (&[Source::ArticleMediaEntities][..], dropped.clone()),
            (&[Source::ArticleCoverMedia][..], dropped),
            (
                &[Source::ArticleMediaEntities, Source::ArticleCoverMedia][..],
                allow(),
            ),
        ] {
            let mut world = InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .user(10, author.clone())
                .viewer(
                    VIEWER,
                    GizmoduckUser {
                        user_id: VIEWER,
                        extended_profile: Some(ExtendedProfile {
                            age_in_years: Some(17),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )
                .tweet_features(1, article_tweet(70))
                .tweet_features(2, article_tweet(70))
                .tweet_features(3, article_tweet(80))
                .lifecycle(70, ArticleLifecycle::Published)
                .lifecycle(80, ArticleLifecycle::Published)
                .article_media_entities(70, true)
                .article_cover_media(70, true);
            for &source in failing {
                world = world.fail_key(source, 70);
            }
            let scenario = executor(world, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
            let hydrated = scenario
                .hydrate(&[raw(1, None), raw(2, None), raw(3, None), raw(4, None)])
                .await;
            assert_eq!(
                hydrated.failed_nodes,
                [
                    Ok(Hydrators::empty()),
                    Ok(Hydrators::empty()),
                    Ok(Hydrators::empty()),
                    Ok(Hydrators::empty())
                ],
                "{failing:?}"
            );
            let evaluations = scenario
                .evaluate(&[raw(1, None), raw(2, None)])
                .await
                .evaluations();
            let expected = Evaluation::complete(verdict);
            assert_eq!(evaluations, [expected.clone(), expected], "{failing:?}");
        }
    }
}
