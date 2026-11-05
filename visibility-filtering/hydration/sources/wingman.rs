use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::{Edge, EdgeKey};
use std::collections::HashMap;
use wingman_client::Exists;
#[cfg(test)]
use {super::InMemoryCall, crate::hydration::hydrator::UserPair, std::collections::HashSet};

impl SourceFetch for source::Wingman {
    const BACKEND_LABEL: &'static str = "wingman";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = Some(Edge::SecondDegree);
    type Key = EdgeKey;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let answers = backends
            .wingman
            .batch_exists_intersect(viewer_id, input.keys)
            .await;
        let answers: HashMap<_, _> = input
            .keys
            .iter()
            .copied()
            .zip(answers.into_iter().flatten())
            .map(|(root, answer)| {
                let answer = match answer {
                    Exists::Found => Ok(Some(true)),
                    Exists::NotFound => Ok(Some(false)),
                    Exists::Incomplete | Exists::ItemError => Err(answer),
                };
                (root, answer)
            })
            .collect();
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            answers,
        ))
    }

    #[cfg(test)]
    type World = HashSet<UserPair>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.viewer_edges(input, world)
    }
}

#[cfg(test)]
mod tests {
    use crate::clients::socialgraph_client::Graph;
    use crate::hydration::execute::tests::{executor, VIEWER};
    use crate::hydration::sources::{control, Fault, InMemorySources, Source};
    use crate::hydration::Hydrator;
    use crate::rules::SafetyLevel;
    use crate::scenario::{raw, InRequestOrder};
    use xai_core_entities::entities::ConversationControlArm;

    #[tokio::test]
    async fn wingman_asks_once_for_my_network_roots_the_followed_by_edge_answered_no() {
        use ConversationControlArm::{Community, MyNetwork};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .tweet(5, 10)
                .tweet(6, 10)
                .authors(&[10])
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(MyNetwork, 60, &[]))
                .control(4, control(MyNetwork, 60, &[]))
                .control(5, control(MyNetwork, 70, &[]))
                .control(6, control(Community, 80, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .second_degree_path(60, VIEWER)
        };
        let raw = [1, 2, 3, 4, 5, 6].map(|id| raw(id, None));
        let second_degree = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| c.edges.contains(Hydrator::RootFollowsViewerSecondDegree))
                .collect::<Vec<_>>()
        };
        let level = SafetyLevel::TimelineHomeHydration;

        let healthy = world();
        let scenario = executor(healthy, level, Some(VIEWER));
        let healthy = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(healthy.keys(Source::Wingman), [vec![60, 70]]);
        assert_eq!(
            second_degree(&hydrated),
            [false, false, true, true, false, false]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fault(Source::Wingman, Fault::Fails);
        let hydrated = executor(failed, level, Some(VIEWER)).hydrate(&raw).await;
        assert_eq!(second_degree(&hydrated), [false; 6]);
        assert!(hydrated.failed_ids.is_empty());

        for (sources, viewer_id, raw) in [
            (world().fail_graph(Graph::Follows), Some(VIEWER), &raw[..]),
            (world(), None, &raw[..]),
            (world(), Some(VIEWER), &raw[..2]),
        ] {
            let scenario = executor(sources, level, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(raw).await;
            assert!(sources.keys(Source::Wingman).is_empty());
            assert_eq!(second_degree(&hydrated), vec![false; raw.len()]);
        }
    }
}
