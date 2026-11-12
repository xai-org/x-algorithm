use super::{source, Backends, Expired, Input, SourceFetch};
use crate::clients::user_location_client;
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::EdgeKey;
use tracing::warn;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashSet};

impl SourceFetch for source::UserLocation {
    const DECIDES_EXISTENCE: bool = false;
    type Key = EdgeKey;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let places = user_location_client::places(&backends.strato, viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "Geoduck userLocation lookup failed"));
        Ok(HydrationBatch::from_hydrated(
            input
                .keys
                .iter()
                .map(|&place| {
                    let outside = places.as_ref().map(|places| Some(!places.contains(&place)));
                    (place, Hydrated::from(outside))
                })
                .collect(),
        ))
    }

    #[cfg(test)]
    type World = HashSet<u64>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed_with(input.keys, |place| Some(!world.contains(&place)))
    }
}
