use super::{source, Backends, Expired, Input, SourceFetch};
use crate::clients::trusted_friends_client;
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::EdgeKey;
#[cfg(test)]
use {super::InMemoryCall, crate::hydration::hydrator::UserPair, std::collections::HashSet};

impl SourceFetch for source::TrustedFriends {
    const DECIDES_EXISTENCE: bool = false;
    type Key = EdgeKey;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let answers = trusted_friends_client::batch_is_member_or_owner(
            &backends.strato,
            viewer_id,
            input.keys,
        )
        .await;
        Ok(HydrationBatch::from_results_with(
            input.keys.iter().copied(),
            answers,
            |answer| Hydrated::from(answer.map(Some)),
        ))
    }

    #[cfg(test)]
    type World = HashSet<UserPair>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.viewer_edges(input, world)
    }
}
