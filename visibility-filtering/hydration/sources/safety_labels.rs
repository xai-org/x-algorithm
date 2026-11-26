use super::{source, Backends, Batches, CallContext, Expired, Input, SourceFetch};
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::fetcher::Fetcher;
use crate::hydration::hydrator::Edge;
use crate::hydration::store::{CallRequest, Lands};
use std::slice;
use std::sync::Arc;
use xai_safety_label_source::{FailureKind, LabelResults};
use xai_visibility_filtering_proto as vf_pb;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashMap};

pub(crate) struct LabelReply {
    batch: RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>,
    stale: Vec<(u64, Arc<vf_pb::SafetyLabelMap>)>,
    ran_out: bool,
}

impl From<RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>> for LabelReply {
    fn from(batch: RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>) -> Self {
        Self {
            batch,
            stale: Vec::new(),
            ran_out: false,
        }
    }
}

impl AsRef<[RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>]> for LabelReply {
    fn as_ref(&self) -> &[RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>] {
        slice::from_ref(&self.batch)
    }
}

impl Batches for LabelReply {
    type Value = Arc<vf_pb::SafetyLabelMap>;

    fn timed_out(failed: RawHydrationBatch<Self::Value>, _: usize) -> Self {
        Self::from(failed)
    }

    fn ran_out(&self) -> bool {
        self.ran_out
    }
}

impl Lands<u64> for LabelReply {
    fn land(self, fetcher: &mut Fetcher<Self::Value>, _: &CallRequest<'_>, _: Option<Edge>) {
        fetcher.land(self.batch);
    }
}

impl SourceFetch for source::SafetyLabels {
    const DECIDES_EXISTENCE: bool = false;
    type Key = u64;
    type Reply = LabelReply;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let LabelResults { labels, stale } = backends
            .safety_labels
            .get_marking_stale(input.keys, xai_x_rpc::current_deadline())
            .await;
        let ran_out = !labels.is_empty()
            && labels.values().all(
                |result| matches!(result, Err(failure) if failure.kind == FailureKind::Deadline),
            );
        Ok(LabelReply {
            batch: HydrationBatch::from_results_with(
                input.keys.iter().copied(),
                labels,
                |labels| Hydrated::from(labels.map(Some)),
            ),
            stale,
            ran_out,
        })
    }

    fn fall_back(_: &CallContext<'_>, reply: Self::Reply) -> Self::Reply {
        if reply.stale.is_empty() {
            return reply;
        }
        let mut hydrated = reply.batch.into_hydrated();
        hydrated.extend(
            reply
                .stale
                .into_iter()
                .map(|(tweet_id, labels)| (tweet_id, Hydrated::Found(labels))),
        );
        LabelReply::from(HydrationBatch::from_hydrated(hydrated))
    }

    #[cfg(test)]
    type World = HashMap<u64, Arc<vf_pb::SafetyLabelMap>>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        LabelReply::from(call.keyed(input.keys, world))
    }
}
