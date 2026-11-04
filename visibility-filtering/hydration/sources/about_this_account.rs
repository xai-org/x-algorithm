use super::{one_viewer_batch, source, Backends, CallContext, Expired, Input, SourceFetch};
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::execute::ViewerCountry;
use crate::hydration::hydrator::Edge;
use crate::hydration::metrics::record_viewer_country;
use std::sync::Arc;
use tracing::warn;
#[cfg(test)]
use {super::InMemoryCall, std::collections::HashMap};

impl SourceFetch for source::ViewerCountry {
    const BACKEND_LABEL: &'static str = "about_this_account";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = true;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<Arc<str>>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let Some(viewer_id) = input.viewer_id else {
            return Ok(HydrationBatch::default());
        };
        let country = backends
            .about_this_account
            .tfe_top_country(viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "tfe_top_country lookup failed"));
        Ok(one_viewer_batch(
            viewer_id,
            country.map(|country| country.map(Arc::from)),
        ))
    }

    fn count(context: &CallContext<'_>, input: &Input<'_>, country: Self::Reply) -> Self::Reply {
        let result = match input
            .viewer_id
            .and_then(|viewer_id| country.hydrated(&viewer_id))
        {
            Some(Hydrated::Found(_)) => ViewerCountry::Found,
            Some(Hydrated::NotFound) => ViewerCountry::NoRow,
            _ => ViewerCountry::Failed,
        };
        record_viewer_country(context.metrics, result.into());
        country
    }

    #[cfg(test)]
    type World = HashMap<u64, Arc<str>>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.viewer_id.as_slice(), world)
    }
}

#[cfg(test)]
mod tests {
    use crate::hydration::execute::tests::{executor, VIEWER};
    use crate::hydration::sources::{control, InMemorySources, Source};
    use crate::rules::SafetyLevel;
    use crate::scenario::{raw, InRequestOrder};
    use xai_core_entities::entities::ConversationControlArm;

    #[tokio::test]
    async fn one_country_lookup_reaches_every_co_tweet_that_needs_it() {
        use ConversationControlArm::Co;
        let raw = [raw(1, None), raw(2, None)];
        let world = |countries: &[&str]| {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .authors(&[10])
                .control(1, control(Co, 30, countries))
                .control(2, control(Co, 30, countries))
                .country(VIEWER, "us")
        };
        let country = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    c.post
                        .conversation_control
                        .as_ref()
                        .unwrap()
                        .viewer_country
                        .as_deref()
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        };

        let listed = world(&["us"]);
        let scenario = executor(listed, SafetyLevel::TimelineHomeHydration, Some(VIEWER));
        let listed = scenario.sources();
        let hydrated = scenario.hydrate(&raw).await;
        assert_eq!(listed.keys(Source::ViewerCountry), [vec![VIEWER]]);
        assert_eq!(
            country(&hydrated),
            [Some("us".to_owned()), Some("us".to_owned())]
        );

        for (sources, viewer_id) in [(world(&[]), Some(VIEWER)), (world(&["us"]), None)] {
            let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, viewer_id);
            let sources = scenario.sources();
            let hydrated = scenario.hydrate(&raw).await;
            assert!(sources.keys(Source::ViewerCountry).is_empty());
            assert_eq!(country(&hydrated), [None, None]);
        }
    }
}
