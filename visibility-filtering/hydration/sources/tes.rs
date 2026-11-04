use super::{source, Backends, CallContext, Expired, Input, SourceFetch};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::decode::tweet::{pure_core, PureCoreColumn, TweetRowColumn};
use crate::hydration::execute::fall_back;
use crate::hydration::hydrator::Edge;
use crate::hydration::metrics::record_tweet_states;
use crate::hydration::tweet_source::{decode_tweet, decode_tweet_state, TweetState};
use crate::models::{PureCore, TweetFeatures};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use xai_core_entities::entities::{ConversationControl, QuotedTweet};
use xai_strato::Bytes;
#[cfg(test)]
use {super::InMemoryCall, xai_core_entities::entities::PureCoreData};

impl SourceFetch for source::TesPureCoreData {
    const BACKEND_LABEL: &'static str = "tes";
    const DECIDES_EXISTENCE: bool = true;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<PureCore>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let cores = backends.tes.get_tweet_core_datas(input.keys.to_vec()).await;
        let cores = HydrationBatch::from_results(input.keys.iter().copied(), cores);
        Ok(cores.map(|core| pure_core(&core)))
    }

    fn fall_back(context: &CallContext<'_>, cores: Self::Reply) -> Self::Reply {
        fall_back::<PureCoreColumn>(context.metrics.plain(), context.tweet_cache, cores)
    }

    #[cfg(test)]
    type World = HashMap<u64, PureCoreData>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world).map(|core| pure_core(&core))
    }
}

impl SourceFetch for source::TesTweetForVisibility {
    const BACKEND_LABEL: &'static str = "tes";
    const DECIDES_EXISTENCE: bool = true;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<TweetFeatures>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let values = backends.tweets.get_tweet_values(input.keys).await;
        Ok(tweet_batch(input.keys, values, decode_tweet))
    }

    fn fall_back(context: &CallContext<'_>, tweets: Self::Reply) -> Self::Reply {
        fall_back::<TweetRowColumn>(context.metrics.plain(), context.tweet_cache, tweets)
    }

    #[cfg(test)]
    type World = HashMap<u64, TweetFeatures>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

impl SourceFetch for source::TesConversationControl {
    const BACKEND_LABEL: &'static str = "tes";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<ConversationControl>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let controls = backends
            .tes
            .get_conversation_controls(input.keys.to_vec())
            .await;
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            controls,
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, ConversationControl>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

impl SourceFetch for source::TesQuotedTweet {
    const BACKEND_LABEL: &'static str = "tes";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<QuotedTweet>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let quoted = backends.tes.get_quoted_tweets(input.keys.to_vec()).await;
        Ok(HydrationBatch::from_results(
            input.keys.iter().copied(),
            quoted,
        ))
    }

    #[cfg(test)]
    type World = HashMap<u64, QuotedTweet>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

impl SourceFetch for source::TesTweetState {
    const BACKEND_LABEL: &'static str = "tes";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = u64;
    type Reply = RawHydrationBatch<TweetState>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let values = backends.tweets.get_tweet_states(input.keys).await;
        Ok(tweet_batch(input.keys, values, decode_tweet_state))
    }

    fn count(context: &CallContext<'_>, _: &Input<'_>, states: Self::Reply) -> Self::Reply {
        record_tweet_states(context.metrics, states)
    }

    #[cfg(test)]
    type World = HashMap<u64, TweetState>;

    #[cfg(test)]
    fn in_memory(world: &Self::World, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        call.keyed(input.keys, world)
    }
}

fn tweet_batch<V>(
    tweet_ids: &[u64],
    values: HashMap<u64, anyhow::Result<Bytes>>,
    decode: fn(&[u8]) -> anyhow::Result<Option<V>>,
) -> RawHydrationBatch<V> {
    let tweets: FxHashMap<_, _> = values
        .into_iter()
        .map(|(id, value)| (id, value.and_then(|bytes| decode(&bytes))))
        .collect();
    HydrationBatch::from_results(tweet_ids.iter().copied(), tweets)
}

#[cfg(test)]
mod tests {
    use crate::hydration::decode::tweet::tweet_fallback_cache;
    use crate::hydration::execute::tests::{executor, tweet_failed, VIEWER};
    use crate::hydration::fallback_cache::FallbackCaches;
    use crate::hydration::sources::{Fault, InMemorySources, Source};
    use crate::hydration::{Cause, Hydrators, Lookup, Relation, Unresolved};
    use crate::models::{TweetFeatures, TweetId};
    use crate::rules::SafetyLevel;
    use crate::scenario::raw;
    use std::collections::HashMap;
    use xai_core_entities::entities::PureCoreData;

    #[tokio::test]
    async fn a_conversation_root_lookup_reads_and_writes_the_tweet_fallback_cache() {
        let nested = |author_id, conversation_id| PureCoreData {
            author_id,
            conversation_id: Some(conversation_id),
            in_reply_to_tweet_id: Some(conversation_id + 1),
            in_reply_to_user_id: Some(99),
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .pure_core(1, nested(10, 100))
            .pure_core(2, nested(11, 200))
            .tweet(100, 30)
            .tweet(200, 40)
            .authors(&[10, 11, 30, 40]);
        let caches = FallbackCaches {
            tweets: Some(tweet_fallback_cache(8)),
            authors: None,
        };
        let scenario = executor(sources, SafetyLevel::TimelineHomeHydration, Some(VIEWER))
            .fallback_caches(caches);
        let first = scenario
            .hydrate(&[raw(1, None), raw(2, None), raw(200, None)])
            .await;
        assert_eq!(first.failed_nodes, [Ok(Hydrators::empty()); 3]);

        let sources = scenario.sources();
        sources.break_source(Source::TesPureCoreData, Fault::Fails);
        sources.break_source(
            (Relation::ConversationRoot, Source::TesPureCoreData),
            Fault::Fails,
        );
        let second = scenario.hydrate(&[raw(100, None), raw(2, None)]).await;
        assert_eq!(second.failed_nodes, [Ok(Hydrators::empty()); 2]);
        assert_eq!(
            sources.keys((Relation::ConversationRoot, Source::TesPureCoreData)),
            [vec![100], vec![200]]
        );
    }

    #[tokio::test]
    async fn the_tweet_cache_serves_the_last_known_core_when_the_call_fails() {
        let sources = InMemorySources::default().tweet(1, 10).authors(&[10]);
        let raw = [raw(1, None)];
        let scenario =
            executor(sources, SafetyLevel::TimelineHome, None).fallback_caches(FallbackCaches {
                tweets: Some(tweet_fallback_cache(8)),
                authors: None,
            });
        let sources = scenario.sources();
        let first = scenario.hydrate(&raw).await;
        assert_eq!(first.candidates[0].author.id, 10);

        sources.break_source(Source::TesPureCoreData, Fault::Fails);
        let second = scenario.hydrate(&raw).await;
        assert_eq!(second.candidates[0].author.id, 10);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_pure_core_not_found_is_not_served_when_the_call_fails() {
        let sources = InMemorySources::default().tweet_features(1, TweetFeatures::default());
        let raw = [raw(1, Some(10))];
        let scenario =
            executor(sources, SafetyLevel::TimelineHome, None).fallback_caches(FallbackCaches {
                tweets: Some(tweet_fallback_cache(8)),
                authors: None,
            });
        let sources = scenario.sources();
        let first = scenario.hydrate(&raw).await;
        assert_eq!(
            first.unresolved,
            HashMap::from([(
                TweetId(1),
                Unresolved {
                    lookup: Lookup::Tweet,
                    cause: Cause::NotFound,
                }
            )])
        );

        sources.break_source(Source::TesPureCoreData, Fault::Fails);
        let second = scenario.hydrate(&raw).await;
        assert_eq!(
            second.unresolved,
            HashMap::from([(TweetId(1), tweet_failed())])
        );
    }
}
