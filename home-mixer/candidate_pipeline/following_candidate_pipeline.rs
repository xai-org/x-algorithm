use crate::candidate_pipeline::reverse_chron_posts_pipeline::ReverseChronPostsPipeline;
use crate::clients::ad_index_client::{
    AdIndexClient,
    MockAdIndexClient,
    ProdAdIndexClient,
};
use crate::clients::past_request_timestamps_client::{
    MockPastRequestTimestampsClient,
    PastRequestTimestampsClient,
    ProdPastRequestTimestampsClient,
};
use crate::clients::prompts_client::{
    MockPromptsClient,
    ProdPromptsClient,
    PromptsClient,
};
use crate::clients::s2s::{
    S2S_CHAIN_PATH,
    S2S_CRT_PATH,
    S2S_KEY_PATH,
};
use crate::clients::served_history_client::{
    MockServedHistoryClient,
    ProdServedHistoryClient,
    ServedHistoryClient,
};
use crate::clients::who_to_follow_client::{
    MockWhoToFollowClient,
    ProdWhoToFollowClient,
    WhoToFollowClient,
};
use crate::filters::invalid_conversation_module_filter::InvalidConversationModuleFilter;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::query_hydrators::followed_user_ids_query_hydrator::FollowedUserIdsQueryHydrator;
use crate::query_hydrators::past_request_timestamps_query_hydrator::PastRequestTimestampsQueryHydrator;
use crate::query_hydrators::served_history_query_hydrator::ServedHistoryQueryHydrator;
use crate::selectors::FollowingBlenderSelector;
use crate::side_effects::ads_injection_logging_side_effect::AdsInjectionLoggingSideEffect;
use crate::side_effects::client_events_kafka_side_effect::ClientEventsKafkaSideEffect;
use crate::side_effects::publish_seen_ids_to_kafka_side_effect::PublishSeenIdsToKafkaSideEffect;
use crate::side_effects::response_stats_side_effect::ResponseStatsSideEffect;
use crate::side_effects::served_ad_history_cache_side_effect::ServedAdHistoryCacheSideEffect;
use crate::side_effects::served_candidates_kafka_side_effect::ServedCandidatesKafkaSideEffect;
use crate::side_effects::truncate_served_history_side_effect::TruncateServedHistorySideEffect;
use crate::side_effects::update_past_request_timestamps_side_effect::UpdatePastRequestTimestampsSideEffect;
use crate::side_effects::update_served_history_side_effect::UpdateServedHistorySideEffect;
use crate::sources::ads_source::AdsSource;
use crate::sources::prompts_source::PromptsSource;
use crate::sources::reverse_chron_posts_source::ReverseChronPostsSource;
use crate::sources::who_to_follow_source::WhoToFollowSource;
use std::sync::Arc;
use tonic::async_trait;

use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaPublisherClient,
    MockKafkaPublisherClient,
};
use xai_candidate_pipeline::component_library::clients::{
    MockSocialGraphClient,
    SocialGraphClient,
    SocialGraphClientOps,
};
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_home_mixer_proto::FeedItem;

pub struct FollowingCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, FeedItem>>>,
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>>,
    selector: FollowingBlenderSelector,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, FeedItem>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>>,
}

/// All dependencies required to construct the following candidate pipeline.
///
/// Keeping these together makes `build()` easier to read and reduces the
/// chance of accidentally wiring a dependency into the wrong component.
struct FollowingPipelineDeps {
    ad_index_client: Arc<dyn AdIndexClient + Send + Sync>,
    served_history_client: Arc<dyn ServedHistoryClient>,
    past_request_timestamps_client: Arc<dyn PastRequestTimestampsClient>,
    socialgraph_client: Arc<dyn SocialGraphClientOps>,
    reverse_chron_pipeline: Arc<ReverseChronPostsPipeline>,
    who_to_follow_client: Arc<dyn WhoToFollowClient + Send + Sync>,
    prompts_client: Arc<dyn PromptsClient + Send + Sync>,

    ads_injection_logging: AdsInjectionLoggingSideEffect,
    served_ad_history: ServedAdHistoryCacheSideEffect,
    publish_seen_ids: PublishSeenIdsToKafkaSideEffect,
    served_candidates: ServedCandidatesKafkaSideEffect,
    client_events: ClientEventsKafkaSideEffect,
}

impl FollowingCandidatePipeline {
    pub async fn new(datacenter: &str) -> Self {
        let (
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            socialgraph_client,
            reverse_chron_pipeline,
            who_to_follow_client,
            prompts_client,
            ads_injection_logging,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
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
            async {
                Arc::new(
                    SocialGraphClient::new(
                        datacenter,
                        &S2S_CHAIN_PATH,
                        &S2S_CRT_PATH,
                        &S2S_KEY_PATH,
                    )
                    .await
                    .expect("Failed to create flock SocialGraphClient"),
                ) as Arc<dyn SocialGraphClientOps>
            },
            ReverseChronPostsPipeline::new(datacenter),
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
            AdsInjectionLoggingSideEffect::prod(),
            ServedAdHistoryCacheSideEffect::prod(datacenter),
            PublishSeenIdsToKafkaSideEffect::prod(),
            ServedCandidatesKafkaSideEffect::prod(),
            ClientEventsKafkaSideEffect::prod(),
        );

        Self::build(FollowingPipelineDeps {
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            socialgraph_client,
            reverse_chron_pipeline: Arc::new(reverse_chron_pipeline),
            who_to_follow_client,
            prompts_client,
            ads_injection_logging,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
        })
    }

    fn build(deps: FollowingPipelineDeps) -> Self {
        let FollowingPipelineDeps {
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            socialgraph_client,
            reverse_chron_pipeline,
            who_to_follow_client,
            prompts_client,
            ads_injection_logging,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
        } = deps;

        // Query hydration happens before candidate generation.
        let query_hydrators = vec![
            Box::new(
                ServedHistoryQueryHydrator::from_client(
                    Arc::clone(&served_history_client),
                ),
            )
                as Box<dyn QueryHydrator<ScoredPostsQuery>>,
            Box::new(
                PastRequestTimestampsQueryHydrator::new(
                    Arc::clone(&past_request_timestamps_client),
                ),
            ),
            Box::new(FollowedUserIdsQueryHydrator {
                socialgraph_client,
            }),
        ];

        // Candidate sources are intentionally kept in the same order as the
        // existing implementation.
        let sources = vec![
            Box::new(ReverseChronPostsSource::new(reverse_chron_pipeline))
                as Box<dyn Source<ScoredPostsQuery, FeedItem>>,
            Box::new(AdsSource {
                ad_index_client,
            }),
            Box::new(WhoToFollowSource {
                who_to_follow_client,
            }),
            Box::new(PromptsSource {
                prompts_client,
            }),
        ];

        // No candidate hydrators are currently configured.
        let hydrators = Vec::new();

        // Preserve the existing filtering behavior.
        let filters = vec![
            Box::new(InvalidConversationModuleFilter)
                as Box<dyn Filter<ScoredPostsQuery, FeedItem>>,
        ];

        let selector = FollowingBlenderSelector::new();

        // No post-selection filters are currently configured.
        let post_selection_filters = Vec::new();

        // Keep side-effect ordering unchanged.
        //
        // The clients are cloned only where they are genuinely shared by
        // multiple components.
        let side_effects: Arc<
            Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>,
        > = Arc::new(vec![
            Box::new(ads_injection_logging),
            Box::new(served_ad_history),
            Box::new(publish_seen_ids),
            Box::new(served_candidates),
            Box::new(client_events),
            Box::new(ResponseStatsSideEffect),
            Box::new(
                UpdatePastRequestTimestampsSideEffect::new(
                    past_request_timestamps_client,
                ),
            ),
            Box::new(
                UpdateServedHistorySideEffect::new(
                    Arc::clone(&served_history_client),
                ),
            ),
            Box::new(
                TruncateServedHistorySideEffect::new(
                    served_history_client,
                ),
            ),
        ]);

        Self {
            query_hydrators,
            sources,
            hydrators,
            filters,
            selector,
            post_selection_filters,
            side_effects,
        }
    }

    pub async fn mock() -> Self {
        let ad_index_client: Arc<dyn AdIndexClient + Send + Sync> =
            Arc::new(MockAdIndexClient);

        let served_history_client: Arc<dyn ServedHistoryClient> =
            Arc::new(MockServedHistoryClient);

        let past_request_timestamps_client:
            Arc<dyn PastRequestTimestampsClient> =
            Arc::new(MockPastRequestTimestampsClient);

        let socialgraph_client: Arc<dyn SocialGraphClientOps> =
            Arc::new(MockSocialGraphClient);

        let reverse_chron_pipeline =
            Arc::new(ReverseChronPostsPipeline::mock().await);

        let who_to_follow_client:
            Arc<dyn WhoToFollowClient + Send + Sync> =
            Arc::new(MockWhoToFollowClient);

        let prompts_client: Arc<dyn PromptsClient + Send + Sync> =
            Arc::new(MockPromptsClient);

        let mock_kafka =
            Arc::new(MockKafkaPublisherClient)
                as Arc<dyn KafkaPublisherClient>;

        let ads_injection_logging =
            AdsInjectionLoggingSideEffect::new(
                Arc::clone(&mock_kafka),
            );

        let served_ad_history =
            ServedAdHistoryCacheSideEffect::new(
                Arc::new(
                    xai_ad_index_history::InMemoryUserAdHistoryStore::default(),
                ),
            );

        let publish_seen_ids =
            PublishSeenIdsToKafkaSideEffect::new(
                Arc::clone(&mock_kafka),
            );

        let served_candidates =
            ServedCandidatesKafkaSideEffect::new(
                Arc::clone(&mock_kafka),
            );

        let client_events =
            ClientEventsKafkaSideEffect::new(
                Arc::clone(&mock_kafka),
            );

        Self::build(FollowingPipelineDeps {
            ad_index_client,
            served_history_client,
            past_request_timestamps_client,
            socialgraph_client,
            reverse_chron_pipeline,
            who_to_follow_client,
            prompts_client,
            ads_injection_logging,
            served_ad_history,
            publish_seen_ids,
            served_candidates,
            client_events,
        })
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, FeedItem>
    for FollowingCandidatePipeline
{
    fn query_hydrators(
        &self,
    ) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        &self.query_hydrators
    }

    fn sources(
        &self,
    ) -> &[Box<dyn Source<ScoredPostsQuery, FeedItem>>] {
        &self.sources
    }

    fn hydrators(
        &self,
    ) -> &[Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>] {
        &self.hydrators
    }

    fn filters(
        &self,
    ) -> &[Box<dyn Filter<ScoredPostsQuery, FeedItem>>] {
        &self.filters
    }

    fn scorers(
        &self,
    ) -> &[Box<dyn Scorer<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn selector(
        &self,
    ) -> &dyn Selector<ScoredPostsQuery, FeedItem> {
        &self.selector
    }

    fn post_selection_hydrators(
        &self,
    ) -> &[Box<dyn Hydrator<ScoredPostsQuery, FeedItem>>] {
        &[]
    }

    fn post_selection_filters(
        &self,
    ) -> &[Box<dyn Filter<ScoredPostsQuery, FeedItem>>] {
        &self.post_selection_filters
    }

    fn side_effects(
        &self,
    ) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, FeedItem>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        params::FOLLOWING_PIPELINE_RESULT_SIZE
    }
}
