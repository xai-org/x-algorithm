use crate::clients::ad_index_client::{AdIndexClient, MockAdIndexClient, ProdAdIndexClient};
use crate::clients::past_request_timestamps_client::{
    MockPastRequestTimestampsClient, PastRequestTimestampsClient, ProdPastRequestTimestampsClient,
};
use crate::clients::served_history_client::{
    MockServedHistoryClient, ProdServedHistoryClient, ServedHistoryClient,
};
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::query_hydrators::past_request_timestamps_query_hydrator::PastRequestTimestampsQueryHydrator;
use crate::query_hydrators::served_history_query_hydrator::ServedHistoryQueryHydrator;
use crate::scored_posts_server::ScoredPostsServer;
use crate::selectors::BlenderSelector;
use crate::side_effects::ads_injection_logging_side_effect::AdsInjectionLoggingSideEffect;
use crate::side_effects::client_events_kafka_side_effect::ClientEventsKafkaSideEffect;
use crate::side_effects::funnel_trace_side_effect::FunnelTraceSideEffect;
use crate::side_effects::phoenix_request_cache_side_effect::ServedSlateCacheSideEffect;
use crate::side_effects::publish_seen_ids_to_kafka_side_effect::PublishSeenIdsToKafkaSideEffect;
use crate::side_effects::response_stats_side_effect::ResponseStatsSideEffect;
use crate::side_effects::served_ad_history_cache_side_effect::ServedAdHistoryCacheSideEffect;
use crate::side_effects::served_candidates_kafka_side_effect::ServedCandidatesKafkaSideEffect;
use crate::side_effects::served_event_kafka_side_effect::ServedEventKafkaSideEffect;
use crate::side_effects::truncate_served_history_side_effect::TruncateServedHistorySideEffect;
use crate::side_effects::update_past_request_timestamps_side_effect::UpdatePastRequestTimestampsSideEffect;
use crate::side_effects::update_served_history_side_effect::UpdateServedHistorySideEffect;
use crate::sources::ads_source::AdsSource;
use crate::sources::scored_posts_source::ScoredPostsSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaPublisherClient, MockKafkaPublisherClient,
};
use xai_candidate_pipeline::component_library::clients::redis_client::MockRedisClient;
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::FeedItem;

pub struct RankedFollowingCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>>,
    selector: BlenderSelector,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>>,
}

impl RankedFollowingCandidatePipeline {
    pub async fn new(scored_posts_server: Arc<ScoredPostsServer>, datacenter: &str) -> Self {
        let (
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            ads_injection_logging,
            funnel_trace,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_event,
            (phoenix_request_cache_redis_atla_client, phoenix_request_cache_redis_pdxa_client),
        ) = tokio::join!(
            async {
                Arc::new(
                    ProdAdIndexClient::new(datacenter)
                        .await
                        .expect("Failed to create AdIndex client"),
                ) as Arc<dyn AdIndexClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdServedHistoryClient::new(datacenter)
                        .await
                        .expect("Failed to create ServedHistoryClient"),
                ) as Arc<dyn ServedHistoryClient>
            },
            async {
                Arc::new(
                    ProdPastRequestTimestampsClient::new(datacenter)
                        .await
                        .expect("Failed to create PastRequestTimestampsClient"),
                ) as Arc<dyn PastRequestTimestampsClient>
            },
            AdsInjectionLoggingSideEffect::prod(),
            FunnelTraceSideEffect::prod(),
            ServedAdHistoryCacheSideEffect::prod(datacenter),
            PublishSeenIdsToKafkaSideEffect::prod(),
            ServedCandidatesKafkaSideEffect::prod(),
            ClientEventsKafkaSideEffect::prod(),
            ServedEventKafkaSideEffect::prod(),
            super::shared_phoenix_request_cache_redis_clients(),
        );
        let served_slate_cache = ServedSlateCacheSideEffect::new(
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
        );

        Self::build(
            scored_posts_server,
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            ads_injection_logging,
            funnel_trace,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_event,
            served_slate_cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        scored_posts_server: Arc<ScoredPostsServer>,
        ad_index_client: Arc<dyn AdIndexClient + Send + Sync>,
        served_history_client: Arc<dyn ServedHistoryClient>,
        past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient>,
        ads_injection_logging: AdsInjectionLoggingSideEffect,
        funnel_trace: FunnelTraceSideEffect,
        served_ad_history: ServedAdHistoryCacheSideEffect,
        publish_seen_ids: PublishSeenIdsToKafkaSideEffect,
        served_candidates: ServedCandidatesKafkaSideEffect,
        client_events: ClientEventsKafkaSideEffect,
        served_event: ServedEventKafkaSideEffect,
        served_slate_cache: ServedSlateCacheSideEffect,
    ) -> Self {
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            Box::new(ServedHistoryQueryHydrator::from_client(Arc::clone(
                &served_history_client,
            ))),
            Box::new(PastRequestTimestampsQueryHydrator::new(Arc::clone(
                &past_request_timestamps_client,
            ))),
        ];

        let sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>> = vec![
            Box::new(ScoredPostsSource {
                scored_posts_server,
            }),
            Box::new(AdsSource { ad_index_client }),
        ];

        let selector = BlenderSelector::new();

        let side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>> =
            Arc::new(vec![
                Box::new(ads_injection_logging),
                Box::new(funnel_trace),
                Box::new(served_ad_history),
                Box::new(publish_seen_ids),
                Box::new(served_candidates),
                Box::new(client_events),
                Box::new(served_event),
                Box::new(ResponseStatsSideEffect),
                Box::new(UpdatePastRequestTimestampsSideEffect::new(
                    past_request_timestamps_client,
                )),
                Box::new(UpdateServedHistorySideEffect::new(Arc::clone(
                    &served_history_client,
                ))),
                Box::new(TruncateServedHistorySideEffect::new(served_history_client)),
                Box::new(served_slate_cache),
            ]);

        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>> = vec![];

        Self {
            query_hydrators,
            sources,
            selector,
            post_selection_filters,
            side_effects,
        }
    }

    pub async fn mock(scored_posts_server: Arc<ScoredPostsServer>) -> Self {
        let ad_index_client: Arc<dyn AdIndexClient + Send + Sync> = Arc::new(MockAdIndexClient);
        let served_history_client: Arc<dyn ServedHistoryClient> = Arc::new(MockServedHistoryClient);
        let past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient> =
            Arc::new(MockPastRequestTimestampsClient);
        let mock_kafka = Arc::new(MockKafkaPublisherClient) as Arc<dyn KafkaPublisherClient>;
        let ads_injection_logging = AdsInjectionLoggingSideEffect::new(Arc::clone(&mock_kafka));
        let funnel_trace = FunnelTraceSideEffect::new(Arc::clone(&mock_kafka));
        let served_ad_history = ServedAdHistoryCacheSideEffect::new(Arc::new(
            xai_ad_index_history::InMemoryUserAdHistoryStore::default(),
        ));
        let publish_seen_ids = PublishSeenIdsToKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_candidates = ServedCandidatesKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let client_events = ClientEventsKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_event = ServedEventKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_slate_cache = ServedSlateCacheSideEffect::new(
            Arc::new(MockRedisClient::default()),
            Arc::new(MockRedisClient::default()),
        );
        Self::build(
            scored_posts_server,
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            ads_injection_logging,
            funnel_trace,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_event,
            served_slate_cache,
        )
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, FeedItem> for RankedFollowingCandidatePipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        &self.query_hydrators
    }

    fn sources(&self) -> &[Box<dyn Source<ScoredPostsQuery, FeedItem>>] {
        &self.sources
    }

    fn hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn scorers(&self) -> &[Box<dyn Scorer<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn selector(&self) -> &dyn Selector<ScoredPostsQuery, FeedItem> {
        &self.selector
    }

    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, FeedItem>>] {
        &self.post_selection_filters
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        params::RANKED_FOLLOWING_MAX_RESULT_SIZE
    }
}
