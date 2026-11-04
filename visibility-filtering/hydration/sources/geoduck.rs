use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::{Edge, EdgeKey};
use std::collections::HashMap;
use tracing::warn;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashSet};

impl SourceFetch for source::UserLocation {
    const BACKEND_LABEL: &'static str = "geoduck";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = Some(Edge::OutsidePlace);
    type Key = EdgeKey;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let places = backends
            .user_location
            .places(viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "Geoduck userLocation lookup failed"));
        let answers: HashMap<_, _> = input
            .keys
            .iter()
            .map(|&place| {
                let outside = places.as_ref().map(|places| Some(!places.contains(&place)));
                (place, outside)
            })
            .collect();
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            answers,
        ))
    }

    #[cfg(test)]
    type World = HashSet<u64>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed_with(input.keys, |place| Some(!world.contains(&place)))
    }
}
