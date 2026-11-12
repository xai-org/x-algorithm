use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use std::sync::Arc;
use xai_visibility_filtering_proto as vf_pb;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashMap};

impl SourceFetch for source::SafetyLabels {
    const DECIDES_EXISTENCE: bool = false;
    type Key = u64;
    type Reply = RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let labels = backends.safety_labels.get(input.keys).await;
        Ok(HydrationBatch::from_results_with(
            input.keys.iter().copied(),
            labels,
            |labels| Hydrated::from(labels.map(Some)),
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, Arc<vf_pb::SafetyLabelMap>>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}
