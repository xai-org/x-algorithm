use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::Edge;
use crate::models::CommunityModeration;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashMap, std::collections::HashSet};

impl SourceFetch for source::CommunityModeration {
    const BACKEND_LABEL: &'static str = "communities";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<CommunityModeration>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let posts = input.community_posts;
        let moderations = backends.communities.moderations(posts).await;
        Ok(HydrationBatch::from_results(
            posts.iter().map(|post| post.tweet_id),
            moderations,
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, CommunityModeration>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        let tweet_ids: Vec<u64> = input
            .community_posts
            .iter()
            .map(|post| post.tweet_id)
            .collect();
        call.keyed_with(&tweet_ids, |tweet_id| {
            Some(world.get(&tweet_id).copied().unwrap_or_default())
        })
    }
}

impl SourceFetch for source::CommunityModerator {
    const BACKEND_LABEL: &'static str = "communities";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let moderators = backends.communities.moderators(viewer_id, input.keys).await;
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            moderators,
        ))
    }

    #[cfg(test)]
    type World = HashSet<u64>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.held_in(input.keys, world)
    }
}

impl SourceFetch for source::CommunityViewerRemoved {
    const BACKEND_LABEL: &'static str = "communities";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let removals = backends
            .communities
            .viewer_removals(viewer_id, input.keys)
            .await;
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            removals,
        ))
    }

    #[cfg(test)]
    type World = HashSet<u64>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.held_in(input.keys, world)
    }
}

#[cfg(test)]
mod tests {
    use crate::hydration::execute::tests::{executor, HIDDEN, VIEWER};
    use crate::hydration::sources::{Fault, InMemorySources, Source};
    use crate::hydration::{Hydrator, Hydrators};
    use crate::models::{ClientCapability, CommunityModeration};
    use crate::rules::SafetyLevel;
    use crate::scenario::{raw, InRequestOrder};

    #[tokio::test]
    async fn only_others_community_posts_are_looked_up_and_only_moderated_ones_ask_the_viewer() {
        use crate::hydration::community_source::CommunityPost;
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .community_post(1, 500)
                .community_moderation(1, HIDDEN)
                .tweet(2, 10)
                .community_post(2, 501)
                .tweet(3, 10)
                .tweet(4, VIEWER)
                .community_post(4, 500)
                .community_moderation(4, HIDDEN)
                .authors(&[10, VIEWER])
                .moderator_of(500)
        };
        let post = |tweet_id, author_id, community_id| CommunityPost {
            tweet_id,
            author_id,
            community_id,
        };
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let level = SafetyLevel::TimelineHomeHydration;

        let logged_in = world();
        let scenario = executor(logged_in, level, Some(VIEWER));
        let logged_in = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(
            logged_in.community_posts(),
            [post(1, 10, 500), post(2, 10, 501)]
        );
        assert_eq!(logged_in.keys(Source::CommunityModerator), [vec![500]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.post.community_moderation,
                    c.post.viewer_is_community_moderator
                ))
                .collect::<Vec<_>>(),
            [
                (HIDDEN, Some(true)),
                (CommunityModeration::default(), None),
                (CommunityModeration::default(), None),
                (CommunityModeration::default(), None),
            ]
        );

        let logged_out = world();
        let scenario = executor(logged_out, level, None);
        let logged_out = scenario.sources();
        scenario.hydrate(&raw).await;
        assert_eq!(
            logged_out.keys(Source::CommunityModeration),
            [vec![1, 2, 4]]
        );
        assert!(logged_out.keys(Source::CommunityModerator).is_empty());

        let failing = world().fault(Source::CommunityModerator, Fault::Fails);
        let hydrated = executor(failing, level, Some(VIEWER)).hydrate(&raw).await;
        assert_eq!(
            hydrated.candidates[0].post.viewer_is_community_moderator,
            None
        );
    }

    #[tokio::test]
    async fn every_community_post_asks_once_per_community_whether_the_viewer_was_removed() {
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .community_post(1, 500)
                .tweet(2, 10)
                .community_post(2, 501)
                .tweet(3, 10)
                .tweet(4, VIEWER)
                .community_post(4, 500)
                .authors(&[10, VIEWER])
                .removed_from(500)
        };
        let level = SafetyLevel::TimelineHomeHydration;
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let removed = |hydrated: &InRequestOrder| -> Vec<bool> {
            hydrated
                .candidates
                .iter()
                .map(|c| c.post.viewer_is_removed_from_community)
                .collect()
        };

        let logged_in = world();
        let scenario = executor(logged_in, level, Some(VIEWER));
        let logged_in = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(
            logged_in.keys(Source::CommunityViewerRemoved),
            [vec![500, 501]]
        );
        assert_eq!(removed(&hydrated), [true, false, false, true]);
        assert!(hydrated.failed_ids.is_empty());

        let logged_out = world();
        let scenario = executor(logged_out, level, None);
        let logged_out = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert!(logged_out.keys(Source::CommunityViewerRemoved).is_empty());
        assert_eq!(removed(&hydrated), [false; 4]);

        let failing = world().fault(Source::CommunityViewerRemoved, Fault::Fails);
        let hydrated = executor(failing, level, Some(VIEWER)).hydrate(&raw).await;
        assert_eq!(removed(&hydrated), [false; 4]);
        assert_eq!(
            hydrated.failed_nodes,
            [
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
                Ok(Hydrators::empty()),
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
            ]
        );

        let without_limits =
            executor(world(), level, Some(VIEWER)).client(ClientCapability::default());
        let hydrated = without_limits.hydrate(&raw).await;
        let removed_keys = without_limits
            .sources()
            .keys(Source::CommunityViewerRemoved);
        assert!(removed_keys.is_empty());
        assert_eq!(removed(&hydrated), [false; 4]);
    }
}
