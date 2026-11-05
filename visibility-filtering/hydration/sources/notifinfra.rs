use super::{held_by, source, Backends, Expired, Input, SourceFetch};
use crate::hydration::batch::RawHydrationBatch;
use crate::hydration::hydrator::Edge;
use tracing::warn;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashSet};

impl SourceFetch for source::ActorEmails {
    const BACKEND_LABEL: &'static str = "notifinfra";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let confirmed = backends
            .user_devices
            .confirmed_emails(input.keys)
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
    const BACKEND_LABEL: &'static str = "notifinfra";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<bool>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let verified = backends
            .user_devices
            .verified_phones(input.keys)
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
