use super::{source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::{Edge, EdgeKey};
use std::collections::HashMap;
#[cfg(test)]
use {super::InMemoryCall, crate::hydration::hydrator::UserPair, std::collections::HashSet};

impl SourceFetch for source::TrustedFriends {
    const BACKEND_LABEL: &'static str = "trusted_friends";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = Some(Edge::TrustedFriends);
    type Key = EdgeKey;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let answers = backends
            .trusted_friends
            .batch_is_member_or_owner(viewer_id, input.keys)
            .await;
        let answers: HashMap<_, _> = input
            .keys
            .iter()
            .copied()
            .zip(answers)
            .map(|(list_id, answer)| (list_id, answer.map(Some)))
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
