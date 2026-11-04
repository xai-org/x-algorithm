use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::Edge;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use xai_visibility_filtering_proto as vf_pb;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashMap};

impl SourceFetch for source::SafetyLabels {
    const BACKEND_LABEL: &'static str = "safety_labels";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let labels: FxHashMap<_, _> = backends
            .safety_labels
            .get(input.keys)
            .await
            .into_iter()
            .map(|(id, labels)| (id, labels.map(Some)))
            .collect();
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            labels,
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, Arc<vf_pb::SafetyLabelMap>>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}
