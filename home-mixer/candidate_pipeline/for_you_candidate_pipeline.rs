use crate::clients::ad_index_client::{AdIndexClient, MockAdIndexClient, ProdAdIndexClient};
use crate::clients::past_request_timestamps_client::{
    MockPastRequestTimestampsClient, PastRequestTimestampsClient, ProdPastRequestTimestampsClient,
};
use crate::clients::prompts_client::{MockPromptsClient, ProdPromptsClient, PromptsClient};
use crate::clients::served_history_client::{
    MockServedHistoryClient, ProdServedHistoryClient, ServedHistoryClient,
};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::clients::who_to_follow_client::{
    MockWhoToFollowClient, ProdWhoToFollowClient, WhoToFollowClient,
};
use crate::filters::ad_adjacent_served_filter::AdAdjacentServedFilter;
use crate::filters::push_to_home_dedup_filter::PushToHomeDedupFilter;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::query_hydrators::frame_dismissal_query_hydrator::FrameDismissalQueryHydrator;
use crate::query_hydrators::past_request_timestamps_query_hydrator::PastRequestTimestampsQueryHydrator;
use crate::query_hydrators::served_history_query_hydrator::ServedHistoryQueryHydrator;
use crate::scored_posts_server::ScoredPostsServer;
use crate::selectors::BlenderSelector;
use crate::side_effects::ads_injection_logging_side_effect::AdsInjectionLoggingSideEffect;
use crate::side_effects::client_events_kafka_side_effect::ClientEventsKafkaSideEffect;
use crate::side_effects::frame_dismissal_side_effect::FrameDismissalSideEffect;
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
use crate::side_effects::video_carousel_candidates_kafka_side_effect::VideoCarouselCandidatesKafkaSideEffect;
use crate::sources::ads_source::AdsSource;
use crate::sources::feed_survey_source::FeedSurveySource;
use crate::sources::jetfuel_direct_frame_source::JetfuelDirectFrameSource;
use crate::sources::jetfuel_frame_source::JetfuelFrameSource;
use crate::sources::prompts_source::PromptsSource;
use crate::sources::push_to_home_source::PushToHomeSource;
use crate::sources::scored_posts_source::ScoredPostsSource;
use crate::sources::who_to_follow_source::WhoToFollowSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaPublisherClient, MockKafkaPublisherClient,
};
use xai_candidate_pipeline::component_library::clients::redis_client::MockRedisClient;
use xai_candidate_pipeline::component_library::clients::{
    JetfuelClient, MockJetfuelClient, MockReplyMixerClient, MockStratoClient, ProdReplyMixerClient,
    ProdStratoClient, ReplyMixerClient, StratoClient,
};
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::FeedItem;
use xai_x_rpc::wily_lookup_service::ShardCoordinate;

pub struct ForYouCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>>,
    selector: BlenderSelector,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>>,
}

impl ForYouCandidatePipeline {
    pub async fn new(
        scored_posts_server: Arc<ScoredPostsServer>,
        jetfuel_client: Arc<dyn JetfuelClient>,
        shard_coordinate: Option<ShardCoordinate>,
        datacenter: &str,
    ) -> Self {
        let (
            ad_index_client,
            ads_injection_logging,
            funnel_trace,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_event,
            video_carousel_candidates,
            served_history_client,
            who_to_follow_client,
            prompts_client,
            past_request_timestamps_client,
            tes_client,
            reply_mixer_client,
            strato_client,
            (phoenix_request_cache_redis_atla_client, phoenix_request_cache_redis_pdxa_client),
        ) = tokio::join!(
            async {
                Arc::new(
                    ProdAdIndexClient::new(datacenter)
                        .await
                        .expect("Failed to create AdIndex client"),
                ) as Arc<dyn AdIndexClient + Send + Sync>
            },
            AdsInjectionLoggingSideEffect::prod(),
            FunnelTraceSideEffect::prod(),
            ServedAdHistoryCacheSideEffect::prod(datacenter),
            PublishSeenIdsToKafkaSideEffect::prod(),
            ServedCandidatesKafkaSideEffect::prod(),
            ClientEventsKafkaSideEffect::prod(),
            ServedEventKafkaSideEffect::prod(),
            VideoCarouselCandidatesKafkaSideEffect::prod(),
            async {
                Arc::new(
                    ProdServedHistoryClient::new(datacenter)
                        .await
                        .expect("Failed to create ServedHistoryClient"),
                ) as Arc<dyn ServedHistoryClient>
            },
            async {
                Arc::new(
                    ProdWhoToFollowClient::new(datacenter)
                        .await
                        .expect("Failed to create WhoToFollowClient"),
                ) as Arc<dyn WhoToFollowClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPromptsClient::new(datacenter)
                        .await
                        .expect("Failed to create PromptsClient"),
                ) as Arc<dyn PromptsClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPastRequestTimestampsClient::new(datacenter)
                        .await
                        .expect("Failed to create PastRequestTimestampsClient"),
                ) as Arc<dyn PastRequestTimestampsClient>
            },
            async {
                Arc::new(
                    ProdTESClient::new(None, datacenter)
                        .await
                        .expect("Failed to create TES client"),
                ) as Arc<dyn TESClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdReplyMixerClient::new(datacenter)
                        .await
                        .expect("Failed to create ReplyMixer client"),
                ) as Arc<dyn ReplyMixerClient>
            },
            async {
                Arc::new(
                    ProdStratoClient::new(shard_coordinate, datacenter)
                        .await
                        .expect("Failed to create Strato client"),
                ) as Arc<dyn StratoClient + Send + Sync>
            },
            super::shared_phoenix_request_cache_redis_clients(),
        );
        let served_slate_cache = ServedSlateCacheSideEffect::new(
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
        );

        Self::build(
            scored_posts_server,
            strato_client,
            jetfuel_client,
            ad_index_client,
            ads_injection_logging,
            funnel_trace,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_event,
            video_carousel_candidates,
            served_history_client,
            who_to_follow_client,
            prompts_client,
            past_request_timestamps_client,
            tes_client,
            reply_mixer_client,
            served_slate_cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        scored_posts_server: Arc<ScoredPostsServer>,
        strato_client: Arc<dyn StratoClient + Send + Sync>,
        jetfuel_client: Arc<dyn JetfuelClient>,
        ad_index_client: Arc<dyn AdIndexClient + Send + Sync>,
        ads_injection_logging: AdsInjectionLoggingSideEffect,
        funnel_trace: FunnelTraceSideEffect,
        served_ad_history: ServedAdHistoryCacheSideEffect,
        publish_seen_ids: PublishSeenIdsToKafkaSideEffect,
        served_candidates: ServedCandidatesKafkaSideEffect,
        client_events: ClientEventsKafkaSideEffect,
        served_event: ServedEventKafkaSideEffect,
        video_carousel_candidates: VideoCarouselCandidatesKafkaSideEffect,
        served_history_client: Arc<dyn ServedHistoryClient>,
        who_to_follow_client: Arc<dyn WhoToFollowClient + Send + Sync>,
        prompts_client: Arc<dyn PromptsClient + Send + Sync>,
        past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        reply_mixer_client: Arc<dyn ReplyMixerClient>,
        served_slate_cache: ServedSlateCacheSideEffect,
    ) -> Self {
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            Box::new(ServedHistoryQueryHydrator::from_client(Arc::clone(
                &served_history_client,
            ))),
            Box::new(PastRequestTimestampsQueryHydrator::new(Arc::clone(
                &past_request_timestamps_client,
            ))),
            Box::new(FrameDismissalQueryHydrator {
                strato_client: Arc::clone(&strato_client),
            }),
        ];

        let sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>> = vec![
            Box::new(ScoredPostsSource {
                scored_posts_server,
            }),
            Box::new(AdsSource { ad_index_client }),
            Box::new(WhoToFollowSource {
                who_to_follow_client,
            }),
            Box::new(PromptsSource { prompts_client }),
            Box::new(PushToHomeSource {
                tes_client,
                reply_mixer_client,
            }),
            Box::new(JetfuelFrameSource {
                strato_client: Arc::clone(&strato_client),
            }),
            Box::new(JetfuelDirectFrameSource { jetfuel_client }),
            Box::new(FeedSurveySource),
        ];

        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>> =
            vec![Box::new(PushToHomeDedupFilter)];

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
                Box::new(video_carousel_candidates),
                Box::new(ResponseStatsSideEffect),
                Box::new(UpdatePastRequestTimestampsSideEffect::new(
                    past_request_timestamps_client,
                )),
                Box::new(UpdateServedHistorySideEffect::new(Arc::clone(
                    &served_history_client,
                ))),
                Box::new(TruncateServedHistorySideEffect::new(served_history_client)),
                Box::new(served_slate_cache),
                Box::new(FrameDismissalSideEffect { strato_client }),
            ]);

        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>> =
            vec![Box::new(AdAdjacentServedFilter)];

        Self {
            query_hydrators,
            sources,
            filters,
            selector,
            post_selection_filters,
            side_effects,
        }
    }

    pub async fn mock(scored_posts_server: Arc<ScoredPostsServer>) -> Self {
        let ad_index_client: Arc<dyn AdIndexClient + Send + Sync> = Arc::new(MockAdIndexClient);
        let mock_kafka = Arc::new(MockKafkaPublisherClient) as Arc<dyn KafkaPublisherClient>;
        let ads_injection = AdsInjectionLoggingSideEffect::new(Arc::clone(&mock_kafka));
        let funnel_trace = FunnelTraceSideEffect::new(Arc::clone(&mock_kafka));
        let served_ad_history = ServedAdHistoryCacheSideEffect::new(Arc::new(
            xai_ad_index_history::InMemoryUserAdHistoryStore::default(),
        ));
        let publish_seen_ids = PublishSeenIdsToKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_candidates = ServedCandidatesKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let client_events = ClientEventsKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_event = ServedEventKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let video_carousel_candidates =
            VideoCarouselCandidatesKafkaSideEffect::new(Arc::clone(&mock_kafka));
        let served_history_client: Arc<dyn ServedHistoryClient> = Arc::new(MockServedHistoryClient);
        let who_to_follow_client: Arc<dyn WhoToFollowClient + Send + Sync> =
            Arc::new(MockWhoToFollowClient);
        let prompts_client: Arc<dyn PromptsClient + Send + Sync> = Arc::new(MockPromptsClient);
        let past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient> =
            Arc::new(MockPastRequestTimestampsClient);
        let tes_client: Arc<dyn TESClient + Send + Sync> = Arc::new(MockTESClient::default());
        let reply_mixer_client: Arc<dyn ReplyMixerClient> = Arc::new(MockReplyMixerClient);
        let strato_client: Arc<dyn StratoClient + Send + Sync> =
            Arc::new(MockStratoClient::default());
        let jetfuel_client: Arc<dyn JetfuelClient> = Arc::new(MockJetfuelClient::default());
        let served_slate_cache = ServedSlateCacheSideEffect::new(
            Arc::new(MockRedisClient::default()),
            Arc::new(MockRedisClient::default()),
        );
        Self::build(
            scored_posts_server,
            strato_client,
            jetfuel_client,
            ad_index_client,
            ads_injection,
            funnel_trace,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
            served_event,
            video_carousel_candidates,
            served_history_client,
            who_to_follow_client,
            prompts_client,
            past_request_timestamps_client,
            tes_client,
            reply_mixer_client,
            served_slate_cache,
        )
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, FeedItem> for ForYouCandidatePipeline {
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
        &self.filters
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
        params::FOR_YOU_MAX_RESULT_SIZE
    }
}
