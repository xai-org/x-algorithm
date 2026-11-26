use super::{held_by, source, Backends, Expired, Input, SourceFetch};
use crate::clients::user_devices_client;
use crate::hydration::batch::RawHydrationBatch;
use tracing::warn;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashSet};

impl SourceFetch for source::ActorEmails {
    const DECIDES_EXISTENCE: bool = false;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let confirmed = user_devices_client::confirmed_emails(&backends.strato, input.keys)
            .await
            .inspect_err(|error| warn!(%error, "confirmed email lookup failed"));
        Ok(held_by(input.keys, confirmed.as_ref().ok()))
    }

    #[cfg(test)]
    type World = HashSet<u64>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.held_in(input.keys, world)
    }
}

impl SourceFetch for source::ActorPhones {
    const DECIDES_EXISTENCE: bool = false;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let verified = user_devices_client::verified_phones(&backends.strato, input.keys)
            .await
            .inspect_err(|error| warn!(%error, "verified phone lookup failed"));
        Ok(held_by(input.keys, verified.as_ref().ok()))
    }

    #[cfg(test)]
    type World = HashSet<u64>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.held_in(input.keys, world)
    }
}
