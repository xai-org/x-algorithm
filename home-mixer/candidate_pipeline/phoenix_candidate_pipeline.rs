use crate::candidate_hydrators::ads_brand_safety_vf_hydrator::AdsBrandSafetyVfHydrator;
use crate::candidate_hydrators::bidirectional_follow_hydrator::BidirectionalFollowHydrator;
use crate::candidate_hydrators::blocked_by_hydrator::BlockedByHydrator;
use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::candidate_hydrators::engagement_counts_hydrator::EngagementCountsHydrator;
use crate::candidate_hydrators::filtered_topics_hydrator::FilteredTopicsHydrator;
use crate::candidate_hydrators::following_replied_users_hydrator::FollowingRepliedUsersHydrator;
use crate::candidate_hydrators::gizmoduck_hydrator::GizmoduckCandidateHydrator;
use crate::candidate_hydrators::in_network_candidate_hydrator::InNetworkCandidateHydrator;
use crate::candidate_hydrators::language_code_hydrator::LanguageCodeHydrator;
use crate::candidate_hydrators::media_info_hydrator::MediaInfoHydrator;
use crate::candidate_hydrators::mutual_follow_jaccard_hydrator::MutualFollowJaccardHydrator;
use crate::candidate_hydrators::quote_hydrator::QuoteHydrator;
use crate::candidate_hydrators::semantic_id_hydrator::SemanticIdHydrator;
use crate::candidate_hydrators::subscription_hydrator::SubscriptionHydrator;
use crate::candidate_hydrators::topic_feedback_context_hydrator::TopicFeedbackContextHydrator;
use crate::candidate_hydrators::tweet_type_metrics_hydrator::TweetTypeMetricsHydrator;
use crate::candidate_hydrators::vf_candidate_hydrator::VFCandidateHydrator;
use crate::clients::engagement_counts_client::{
    EngagementCountsClient, ProdEngagementCountsClient,
};
use crate::clients::engagement_signals_client::{
    EngagementSignalsClient, MockEngagementSignalsClient, ProdEngagementSignalsClient,
};
use crate::clients::gizmoduck_client::{GizmoduckClient, MockGizmoduckClient, ProdGizmoduckClient};

use crate::clients::impressed_posts_client::ImpressedPostsClient;
use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use crate::clients::simclusters_ann_client::{
    MockSimClustersAnnClient, ProdSimClustersAnnClient, SimClustersAnnClient,
};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::clients::user_action_aggregation_client::{
    MockUserActionAggregationClient, ProdUserActionAggregationClient, UserActionAggregationClient,
};
use crate::clients::vm_ranker_client::{MockVMRankerClient, ProdVMRankerClient, VMRankerClient};
use crate::filters::age_filter::AgeFilter;
use crate::filters::ancillary_vf_filter::AncillaryVFFilter;
use crate::filters::author_socialgraph_filter::AuthorSocialgraphFilter;
use crate::filters::brazil_2026_election_filter::Brazil2026ElectionFilter;
use crate::filters::core_data_hydration_filter::CoreDataHydrationFilter;
use crate::filters::dedup_conversation_filter::DedupConversationFilter;
use crate::filters::drop_duplicates_filter::DropDuplicatesFilter;
use crate::filters::ineligible_subscription_filter::IneligibleSubscriptionFilter;
use crate::filters::inventory_holdout_filter::InventoryHoldoutFilter;
use crate::filters::muted_keyword_filter::MutedKeywordFilter;
use crate::filters::new_user_min_engagement_filter::NewUserMinEngagementFilter;
use crate::filters::oon_nsfw_simclusters_filter::OONNsfwSimclustersFilter;
use crate::filters::oon_retweet_reply_filter::OONRetweetReplyFilter;
use crate::filters::previously_seen_posts_backup_filter::PreviouslySeenPostsBackupFilter;
use crate::filters::previously_seen_posts_filter::PreviouslySeenPostsFilter;
use crate::filters::previously_served_posts_filter::PreviouslyServedPostsFilter;
use crate::filters::retweet_deduplication_filter::RetweetDeduplicationFilter;
use crate::filters::self_tweet_filter::SelfTweetFilter;
use crate::filters::topic_ids_filter::TopicIdsFilter;
use crate::filters::vf_filter::VFFilter;
use crate::filters::video_filter::VideoFilter;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::query_hydrators::blocked_user_ids_query_hydrator::BlockedUserIdsQueryHydrator;
use crate::query_hydrators::cached_posts_query_hydrator::CachedPostsQueryHydrator;
use crate::query_hydrators::explicit_engagement_signals_query_hydrator::ExplicitEngagementSignalsQueryHydrator;
use crate::query_hydrators::followed_grok_topics_query_hydrator::FollowedGrokTopicsQueryHydrator;
use crate::query_hydrators::followed_starter_packs_query_hydrator::FollowedStarterPacksQueryHydrator;
use crate::query_hydrators::followed_user_ids_query_hydrator::FollowedUserIdsQueryHydrator;
use crate::query_hydrators::implicit_engagement_signals_query_hydrator::ImplicitEngagementSignalsQueryHydrator;
use crate::query_hydrators::impressed_posts_query_hydrator::ImpressedPostsQueryHydrator;
use crate::query_hydrators::impression_bloom_filter_query_hydrator::ImpressionBloomFilterQueryHydrator;
use crate::query_hydrators::ip_query_hydrator::IpQueryHydrator;
use crate::query_hydrators::muted_user_ids_query_hydrator::MutedUserIdsQueryHydrator;
use crate::query_hydrators::mutual_follow_query_hydrator::MutualFollowQueryHydrator;
use crate::query_hydrators::retrieval_sequence_query_hydrator::RetrievalSequenceQueryHydrator;
use crate::query_hydrators::scoring_sequence_query_hydrator::ScoringSequenceQueryHydrator;
use crate::query_hydrators::subscribed_user_ids_query_hydrator::SubscribedUserIdsQueryHydrator;
use crate::query_hydrators::user_demographics_query_hydrator::UserDemographicsQueryHydrator;
use crate::query_hydrators::user_inferred_gender_query_hydrator::UserInferredGenderQueryHydrator;
use crate::query_hydrators::user_installed_apps_query_hydrator::UserInstalledAppsQueryHydrator;
use crate::scorers::phoenix_scorer::PhoenixScorer;
use crate::scorers::ranking_scorer::RankingScorer;
use crate::scorers::vm_ranker::VMRanker;
use crate::selectors::TopKScoreSelector;
use crate::side_effects::author_served_metrics_side_effect::AuthorServedMetricsSideEffect;
use crate::side_effects::debug_side_effect::DebugSideEffect;
use crate::side_effects::mutual_follow_stats_side_effect::MutualFollowStatsSideEffect;
use crate::side_effects::phoenix_experiments_side_effect::PhoenixExperimentsSideEffect;
use crate::side_effects::phoenix_request_cache_side_effect::PhoenixRequestCacheSideEffect;
use crate::side_effects::redis_post_candidate_cache_side_effect::RedisPostCandidateCacheSideEffect;
use crate::side_effects::reranking_kafka_side_effect::RerankingKafkaSideEffect;
use crate::side_effects::scored_stats_side_effect::ScoredStatsSideEffect;
use crate::sources::cached_posts_source::CachedPostsSource;
use crate::sources::phoenix_moe_source::PhoenixMOESource;
use crate::sources::phoenix_source::PhoenixSource;
use crate::sources::phoenix_topics_source::PhoenixTopicsSource;
use crate::sources::simclusters_source::SimclustersSource;
use crate::sources::thunder_source::ThunderSource;
use crate::sources::tweet_mixer_source::TweetMixerSource;
use xai_candidate_pipeline::component_library::clients::followed_grok_topics_store_client::{
    FollowedGrokTopicsStoreClient, MockFollowedGrokTopicsStoreClient,
    ProdFollowedGrokTopicsStoreClient,
};
use xai_candidate_pipeline::component_library::clients::followed_starter_packs_store_client::{
    FollowedStarterPacksStoreClient, MockFollowedStarterPacksStoreClient,
    ProdFollowedStarterPacksStoreClient,
};
use xai_candidate_pipeline::component_library::clients::gender_prediction_client::{
    GenderPredictionGrpcClient, MockGenderPredictionGrpcClient, ProdGenderPredictionGrpcClient,
};
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, MockKafkaPublisherClient, ProdKafkaPublisherClient,
    PHOENIX_SCORES_TOPIC, RERANKING_TOPIC,
};
use xai_candidate_pipeline::component_library::clients::media_info_cache_client::{
    MediaInfoCacheClient, MockMediaInfoCacheClient, ProdMediaInfoCacheClient,
};
use xai_candidate_pipeline::component_library::clients::user_demographics_client::{
    MockUserDemographicsClient, ProdUserDemographicsClient, UserDemographicsClient,
};
use xai_candidate_pipeline::component_library::clients::user_inferred_gender_store_client::{
    MockUserInferredGenderStoreClient, ProdUserInferredGenderStoreClient,
    UserInferredGenderStoreClient,
};
use xai_candidate_pipeline::component_library::clients::user_installed_apps_store_client::{
    MockUserInstalledAppsStoreClient, ProdUserInstalledAppsStoreClient,
    UserInstalledAppsStoreClient,
};
use xai_candidate_pipeline::component_library::clients::{MockSidClient, ProdSidClient, SidClient};
use xai_candidate_pipeline::component_library::clients::{
    MockTweetMixerClient, ProdTweetMixerClient, TweetMixerClient,
};

use std::sync::Arc;
use std::time::Duration;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    MockPredictClient, PhoenixPredictionClient, ProdPhoenixPredictionClient,
};
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::{
    MockRetrievalClient, PhoenixRetrievalClient, PhoenixRetrievalCluster,
    ProdPhoenixRetrievalClient,
};
use xai_candidate_pipeline::component_library::clients::redis_client::{
    MockRedisClient, RedisClient,
};
use xai_candidate_pipeline::component_library::clients::{
    ImpressionBloomFilterClient, MockImpressionBloomFilterClient, ProdImpressionBloomFilterClient,
};
use xai_candidate_pipeline::component_library::clients::{
    MockSocialGraphClient, SocialGraphClient, SocialGraphClientOps,
};
use xai_candidate_pipeline::component_library::clients::{
    MockStratoClient, ProdStratoClient, StratoClient,
};
use xai_candidate_pipeline::component_library::clients::{
    ProdThunderCapiClient, ThunderCapiClient, ThunderClient,
};
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_feature_switches::FeatureSwitches;
use xai_geo_ip::GeoIpLocationClient;
use xai_redis_client::{XdsRedisClient, XdsRedisConfig};
use xai_visibility_filtering::tweet_safety_label::{
    MockTweetSafetyLabelClient, ProdTweetSafetyLabelClient, TweetSafetyLabelClient,
};
use xai_visibility_filtering::vf_client::{MockVfClient, StratoVfClient, VfClient, XaiVfClient};
use xai_x_rpc::wily_lookup_service::ShardCoordinate;

pub struct PhoenixCandidatePipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>,
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>>,
    selector: TopKScoreSelector,
    post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>>,
}

impl PhoenixCandidatePipeline {
    pub(crate) async fn build_with_clients(
        user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
        thunder_client: Arc<ThunderClient>,
        thunder_capi_client: Option<Arc<dyn ThunderCapiClient + Send + Sync>>,
        strato_client: Arc<dyn StratoClient + Send + Sync>,
        tweet_mixer_client: Arc<dyn TweetMixerClient>,
        simclusters_ann_client: Arc<dyn SimClustersAnnClient + Send + Sync>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
        gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
        strato_vf_client: Arc<dyn VfClient + Send + Sync>,
        xai_vf_client: Arc<dyn VfClient + Send + Sync>,
        redis_client: Arc<dyn RedisClient + Send + Sync>,
        phoenix_kafka_client: Arc<dyn KafkaPublisherClient>,
        reranking_kafka_client: Arc<dyn KafkaPublisherClient>,
        socialgraph_client: Arc<dyn SocialGraphClientOps>,
        vm_ranker_client: Arc<dyn VMRankerClient>,
        vf_safety_labels_client: Arc<dyn TweetSafetyLabelClient>,
        phoenix_request_cache_redis_atla_client: Arc<dyn RedisClient + Send + Sync>,
        phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient + Send + Sync>,
        impression_bloom_filter_client: Arc<dyn ImpressionBloomFilterClient>,
        ip_client: Arc<GeoIpLocationClient>,
        user_demographics_client: Arc<dyn UserDemographicsClient>,
        user_inferred_gender_store_client: Arc<dyn UserInferredGenderStoreClient>,
        user_inferred_gender_grpc_client: Arc<dyn GenderPredictionGrpcClient>,
        impressed_posts_client: Arc<dyn ImpressedPostsClient>,
        engagement_counts_client: Arc<dyn EngagementCountsClient>,
        followed_grok_topics_client: Arc<dyn FollowedGrokTopicsStoreClient>,
        followed_starter_packs_client: Arc<dyn FollowedStarterPacksStoreClient>,
        user_installed_apps_client: Arc<dyn UserInstalledAppsStoreClient>,
        engagement_signals_client: Arc<dyn EngagementSignalsClient>,
        feature_switches: Arc<FeatureSwitches>,
        phoenix_xds: &super::PhoenixXdsConfig,
        vm_ranker_xds: &super::VmRankerXdsConfig,
        sid_client: Arc<dyn SidClient>,
    ) -> PhoenixCandidatePipeline {
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            Box::new(ScoringSequenceQueryHydrator::new(
                user_action_aggregation_client.clone(),
            )),
            Box::new(RetrievalSequenceQueryHydrator::new(
                user_action_aggregation_client,
            )),
            Box::new(BlockedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(MutedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(FollowedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(SubscribedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(CachedPostsQueryHydrator {
                redis_client: redis_client.clone(),
            }),
            Box::new(MutualFollowQueryHydrator {
                strato_client: strato_client.clone(),
            }),
            Box::new(UserDemographicsQueryHydrator {
                client: user_demographics_client,
            }),
            Box::new(FollowedGrokTopicsQueryHydrator::new(
                followed_grok_topics_client,
            )),
            Box::new(FollowedStarterPacksQueryHydrator::new(
                followed_starter_packs_client,
            )),
            Box::new(UserInstalledAppsQueryHydrator::new(
                user_installed_apps_client,
            )),
            Box::new(ExplicitEngagementSignalsQueryHydrator::new(
                engagement_signals_client.clone(),
            )),
            Box::new(ImplicitEngagementSignalsQueryHydrator::new(
                engagement_signals_client,
            )),
            Box::new(ImpressionBloomFilterQueryHydrator {
                client: impression_bloom_filter_client,
            }),
            Box::new(IpQueryHydrator { client: ip_client }),
            Box::new(UserInferredGenderQueryHydrator::new(
                user_inferred_gender_store_client,
                user_inferred_gender_grpc_client,
            )),
        ];

        let _impressed_posts_hydrator = ImpressedPostsQueryHydrator {
            client: impressed_posts_client,
        };

        let xds_retrieval_client = super::build_phoenix_xds_retrieval_client(phoenix_xds).await;
        let mut retrieval_paths = Vec::new();
        if let Some(xds) = xds_retrieval_client {
            retrieval_paths.push(crate::util::egress::RetrievalPath {
                name: "xDS",
                gate: crate::util::xds::RETRIEVAL_XDS_GATE,
                client: xds,
                config: crate::util::egress::EgressConfig::DEFAULT,
            });
        }
        let retrieval_dispatch = crate::util::egress::RetrievalDispatch {
            prod: phoenix_retrieval_client,
            paths: retrieval_paths,
        };
        let phoenix_source = Box::new(PhoenixSource {
            dispatch: retrieval_dispatch.clone(),
        });
        let phoenix_topics_source = Box::new(PhoenixTopicsSource {
            dispatch: retrieval_dispatch.clone(),
        });
        let phoenix_moe_source = Box::new(PhoenixMOESource {
            dispatch: retrieval_dispatch,
        });
        let thunder_source = Box::new(ThunderSource {
            thunder_client,
            thunder_capi_client,
        });
        let tweet_mixer_source = Box::new(TweetMixerSource { tweet_mixer_client });
        let core_data_hydrator = CoreDataCandidateHydrator::new(tes_client.clone()).await;
        let simclusters_source = Box::new(SimclustersSource::new(
            simclusters_ann_client,
            core_data_hydrator.clone(),
        ));
        let cached_posts_source = Box::new(CachedPostsSource);
        let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> = vec![
            thunder_source,
            tweet_mixer_source,
            simclusters_source,
            phoenix_source,
            phoenix_topics_source,
            phoenix_moe_source,
            cached_posts_source,
        ];

        let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(InNetworkCandidateHydrator),
            Box::new(BidirectionalFollowHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(core_data_hydrator),
            Box::new(QuoteHydrator::new(tes_client.clone(), socialgraph_client.clone()).await),
            Box::new(MediaInfoHydrator::new(media_info_cache_client).await),
            Box::new(SubscriptionHydrator::new(tes_client.clone()).await),
            Box::new(GizmoduckCandidateHydrator::new(gizmoduck_client).await),
            Box::new(BlockedByHydrator::new(socialgraph_client).await),
            Box::new(FilteredTopicsHydrator {
                strato_client: strato_client.clone(),
            }),
            Box::new(LanguageCodeHydrator::new(tes_client.clone()).await),
            Box::new(EngagementCountsHydrator::new(engagement_counts_client).await),
            Box::new(SemanticIdHydrator::new(sid_client)),
        ];

        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(DropDuplicatesFilter),
            Box::new(CoreDataHydrationFilter),
            Box::new(AgeFilter::new(Duration::from_secs(params::MAX_POST_AGE))),
            Box::new(SelfTweetFilter),
            Box::new(OONRetweetReplyFilter),
            Box::new(OONNsfwSimclustersFilter),
            Box::new(RetweetDeduplicationFilter),
            Box::new(IneligibleSubscriptionFilter),
            Box::new(PreviouslySeenPostsFilter),
            Box::new(PreviouslySeenPostsBackupFilter),
            Box::new(PreviouslyServedPostsFilter),
            Box::new(MutedKeywordFilter::new()),
            Box::new(AuthorSocialgraphFilter),
            // Brazil 2026 election filter

            // Application providers that use a recommendation system for users must exclude from the
            // results the channels and profiles reported to the Electoral Court under the terms of
            // § 1º of this article and, except in cases of paid boosting, the content posted on them.

            // https://dadosabertos.tse.jus.br/dataset/candidatos-2026

            // OmarAzizSenador deleted his account at the time this code was written.
            Box::new(Brazil2026ElectionFilter),
            Box::new(VideoFilter),
            Box::new(TopicIdsFilter),
            Box::new(NewUserMinEngagementFilter),
            Box::new(InventoryHoldoutFilter),
        ];

        let xds_client = super::build_phoenix_xds_client(phoenix_xds).await;
        let mut prediction_paths = Vec::new();
        if let Some(xds) = xds_client.as_ref() {
            prediction_paths.push(crate::util::egress::PredictionPath {
                name: "xDS",
                gate: crate::util::xds::XDS_GATE,
                client: Arc::clone(xds),
                config: crate::util::egress::EgressConfig::DEFAULT,
            });
        }
        let phoenix_scorer = Box::new(PhoenixScorer {
            dispatch: crate::util::egress::PredictionDispatch {
                prod: phoenix_client.clone(),
                paths: prediction_paths,
                max_retries_key: "rust_home_mixer_phoenix_xds_max_retries",
                enable_fallback_key: "rust_home_mixer_phoenix_enable_fallback",
            },
        });
        let author_rules = Arc::new(crate::util::author_rules::AuthorRulesEvaluator::new(
            feature_switches,
        ));
        let author_cold_start = crate::scorers::author_cold_start::AuthorColdStart { author_rules };
        let ranking_scorer = Box::new(RankingScorer {
            author_cold_start: author_cold_start.clone(),
        });
        let xds_vm_ranker_client = super::build_vm_ranker_xds_client(vm_ranker_xds).await;
        let vm_ranker = Box::new(VMRanker {
            client: vm_ranker_client,
            xds_client: xds_vm_ranker_client,
            author_cold_start,
        });
        let scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>> =
            vec![phoenix_scorer, ranking_scorer, vm_ranker];

        let selector = TopKScoreSelector;

        let post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(
                VFCandidateHydrator::new(strato_vf_client.clone(), xai_vf_client.clone()).await,
            ),
            Box::new(AdsBrandSafetyVfHydrator {
                client: vf_safety_labels_client,
            }),
            Box::new(TweetTypeMetricsHydrator::new()),
            Box::new(FollowingRepliedUsersHydrator),
            Box::new(MutualFollowJaccardHydrator {
                strato_client: strato_client.clone(),
            }),
            Box::new(TopicFeedbackContextHydrator {
                strato_client: strato_client.clone(),
            }),
        ];

        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(VFFilter),
            Box::new(AncillaryVFFilter),
            Box::new(DedupConversationFilter),
        ];

        let side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> =
            Arc::new(vec![
                Box::new(PhoenixExperimentsSideEffect::new(
                    phoenix_client,
                    xds_client,
                    phoenix_kafka_client,
                )),
                Box::new(RerankingKafkaSideEffect::new(reranking_kafka_client)),
                Box::new(RedisPostCandidateCacheSideEffect::new(redis_client)),
                Box::new(ScoredStatsSideEffect),
                Box::new(AuthorServedMetricsSideEffect),
                Box::new(MutualFollowStatsSideEffect),
                Box::new(DebugSideEffect),
                Box::new(PhoenixRequestCacheSideEffect::new(
                    phoenix_request_cache_redis_atla_client,
                    phoenix_request_cache_redis_pdxa_client,
                )),
            ]);

        PhoenixCandidatePipeline {
            query_hydrators,
            hydrators,
            filters,
            sources,
            scorers,
            selector,
            post_selection_hydrators,
            post_selection_filters,
            side_effects,
        }
    }

    pub async fn prod(
        shard_coordinate: Option<ShardCoordinate>,
        datacenter: &str,
        feature_switches: Arc<FeatureSwitches>,
        phoenix_xds: &super::PhoenixXdsConfig,
        vm_ranker_xds: &super::VmRankerXdsConfig,
    ) -> PhoenixCandidatePipeline {
        let local_cache_eds = format!("");
        let atla_phoenix_cache_eds = "";
        let pdxa_phoenix_cache_eds = "";

        let (
            flock_socialgraph_client,
            user_action_aggregation_client,
            phoenix_client,
            phoenix_retrieval_client,
            thunder_client,
            strato_client,
            tweet_mixer_client,
            simclusters_ann_client,
            tes_client,
            media_info_cache_client,
            gizmoduck_client,
            strato_vf_client,
            xai_vf_client,
            redis_client,
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
            phoenix_kafka_client,
            reranking_kafka_client,
            vm_ranker_client,
            vf_safety_labels_client,
            impression_bloom_filter_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            impressed_posts_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
            user_installed_apps_client,
            engagement_signals_client,
            engagement_counts_client_impl,
            sid_client,
            thunder_capi_client,
        ) = tokio::join!(
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
            async {
                Arc::new(
                    ProdUserActionAggregationClient::new()
                        .await
                        .expect("Failed to create User Action Aggregation client"),
                ) as Arc<dyn UserActionAggregationClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPhoenixPredictionClient::new()
                        .await
                        .expect("Failed to create Phoenix prediction client"),
                ) as Arc<dyn PhoenixPredictionClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdPhoenixRetrievalClient::new(Some((
                        PhoenixRetrievalCluster::Experiment1Fou,
                        PhoenixRetrievalCluster::Experiment2Fou,
                    )))
                    .await
                    .expect("Failed to create Phoenix retrieval client"),
                ) as Arc<dyn PhoenixRetrievalClient + Send + Sync>
            },
            async { Arc::new(ThunderClient::new().await) },
            async {
                Arc::new(
                    ProdStratoClient::new(shard_coordinate, datacenter)
                        .await
                        .expect("Failed to create Strato client"),
                ) as Arc<dyn StratoClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdTweetMixerClient::new(datacenter)
                        .await
                        .expect("Failed to create TweetMixer client"),
                ) as Arc<dyn TweetMixerClient>
            },
            async {
                Arc::new(
                    ProdSimClustersAnnClient::new(datacenter)
                        .await
                        .expect("Failed to create SimClusters ANN client"),
                ) as Arc<dyn SimClustersAnnClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdTESClient::new(shard_coordinate, datacenter)
                        .await
                        .expect("Failed to create TES client"),
                ) as Arc<dyn TESClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdMediaInfoCacheClient::new(datacenter, "home-mixer")
                        .await
                        .expect("Failed to create MediaInfoCacheClient"),
                ) as Arc<dyn MediaInfoCacheClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdGizmoduckClient::new(
                        shard_coordinate,
                        datacenter,
                        Some("home-mixer.prod".to_string()),
                    )
                    .await
                    .expect("Failed to create Gizmoduck client"),
                ) as Arc<dyn GizmoduckClient + Send + Sync>
            },
            async {
                Arc::new(
                    StratoVfClient::new(
                        S2S_CHAIN_PATH.clone(),
                        S2S_CRT_PATH.clone(),
                        S2S_KEY_PATH.clone(),
                        "home-mixer.prod".to_string(),
                        datacenter.to_string(),
                    )
                    .await
                    .expect("Failed to create VF client"),
                ) as Arc<dyn VfClient + Send + Sync>
            },
            async {
                Arc::new(
                    XaiVfClient::connect(datacenter)
                        .await
                        .expect("Failed to create XAI VF client"),
                ) as Arc<dyn VfClient + Send + Sync>
            },
            async {
                Arc::new(
                    XdsRedisClient::new(XdsRedisConfig {
                        eds_resource_name: local_cache_eds.clone(),
                    })
                    .await
                    .expect("Failed to create xDS Redis client for local cache"),
                ) as Arc<dyn RedisClient + Send + Sync>
            },
            async {
                Arc::new(
                    XdsRedisClient::new(XdsRedisConfig {
                        eds_resource_name: atla_phoenix_cache_eds.into(),
                    })
                    .await
                    .expect("Failed to create xDS Redis client for atla phoenix cache"),
                ) as Arc<dyn RedisClient + Send + Sync>
            },
            async {
                Arc::new(
                    XdsRedisClient::new(XdsRedisConfig {
                        eds_resource_name: pdxa_phoenix_cache_eds.into(),
                    })
                    .await
                    .expect("Failed to create xDS Redis client for pdxa phoenix cache"),
                ) as Arc<dyn RedisClient + Send + Sync>
            },
            async {
                Arc::new(
                    ProdKafkaPublisherClient::new(PHOENIX_SCORES_TOPIC, KafkaCluster::Aiml).await,
                ) as Arc<dyn KafkaPublisherClient>
            },
            async {
                Arc::new(
                    ProdKafkaPublisherClient::new(RERANKING_TOPIC, KafkaCluster::Phoenix).await,
                ) as Arc<dyn KafkaPublisherClient>
            },
            async {
                Arc::new(
                    ProdVMRankerClient::new()
                        .await
                        .expect("Failed to create VMRanker client"),
                ) as Arc<dyn VMRankerClient>
            },
            async {
                Arc::new(
                    ProdTweetSafetyLabelClient::new(datacenter)
                        .await
                        .expect("Failed to create VF SafetyLabels client")
                        .with_timeout_ms(500)
                        .with_max_batch_size(50),
                ) as Arc<dyn TweetSafetyLabelClient>
            },
            async {
                Arc::new(
                    ProdImpressionBloomFilterClient::new(datacenter)
                        .await
                        .expect("Failed to create ImpressionBloomFilter client"),
                ) as Arc<dyn ImpressionBloomFilterClient>
            },
            async {
                Arc::new(
                    GeoIpLocationClient::new(
                        &S2S_CHAIN_PATH,
                        &S2S_CRT_PATH,
                        &S2S_KEY_PATH,
                        datacenter,
                    )
                    .await
                    .expect("Failed to create GeoIpLocationClient"),
                )
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdUserDemographicsClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create UserDemographics client"),
                ) as Arc<dyn UserDemographicsClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdUserInferredGenderStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create UserInferredGenderStore client"),
                ) as Arc<dyn UserInferredGenderStoreClient>
            },
            async {
                Arc::new(
                    ProdGenderPredictionGrpcClient::new()
                        .await
                        .expect("Failed to create GenderPredictionGrpcClient"),
                ) as Arc<dyn GenderPredictionGrpcClient>
            },
            async {
                Arc::new(
                    crate::clients::impressed_posts_client::ProdImpressedPostsClient::new(
                        datacenter,
                    )
                    .await
                    .expect("Failed to create ImpressedPosts client"),
                ) as Arc<dyn ImpressedPostsClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdFollowedGrokTopicsStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create FollowedGrokTopicsStore client"),
                ) as Arc<dyn FollowedGrokTopicsStoreClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdFollowedStarterPacksStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create FollowedStarterPacksStore client"),
                ) as Arc<dyn FollowedStarterPacksStoreClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdUserInstalledAppsStoreClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create UserInstalledAppsStore client"),
                ) as Arc<dyn UserInstalledAppsStoreClient>
            },
            async {
                let s2s = xai_manhattan::s2s::S2sConfig {
                    client_cert_path: S2S_CRT_PATH.clone(),
                    client_key_path: S2S_KEY_PATH.clone(),
                    ca_cert_path: S2S_CHAIN_PATH.clone(),
                };
                Arc::new(
                    ProdEngagementSignalsClient::new(datacenter, s2s)
                        .await
                        .expect("Failed to create EngagementSignals client"),
                ) as Arc<dyn EngagementSignalsClient>
            },
            async {
                Arc::new(
                    ProdEngagementCountsClient::new(datacenter)
                        .await
                        .expect("Failed to create EngagementCounts client"),
                )
            },
            async { Arc::new(ProdSidClient::new()) as Arc<dyn SidClient> },
            async {
                match ProdThunderCapiClient::new(datacenter).await {
                    Ok(c) => Some(Arc::new(c) as Arc<dyn ThunderCapiClient + Send + Sync>),
                    Err(e) => {
                        tracing::warn!(error = %e, "ThunderCapiClient build failed; using proxy path");
                        None
                    }
                }
            },
        );

        let engagement_counts_client: Arc<dyn EngagementCountsClient> =
            engagement_counts_client_impl;

        PhoenixCandidatePipeline::build_with_clients(
            user_action_aggregation_client,
            phoenix_client,
            phoenix_retrieval_client,
            thunder_client,
            thunder_capi_client,
            strato_client,
            tweet_mixer_client,
            simclusters_ann_client,
            tes_client,
            media_info_cache_client,
            gizmoduck_client,
            strato_vf_client,
            xai_vf_client,
            redis_client,
            phoenix_kafka_client,
            reranking_kafka_client,
            flock_socialgraph_client,
            vm_ranker_client,
            vf_safety_labels_client,
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
            impression_bloom_filter_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            impressed_posts_client,
            engagement_counts_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
            user_installed_apps_client,
            engagement_signals_client,
            feature_switches,
            phoenix_xds,
            vm_ranker_xds,
            sid_client,
        )
        .await
    }

    pub async fn mock() -> PhoenixCandidatePipeline {
        let user_action_aggregation_client = Arc::new(MockUserActionAggregationClient);
        let phoenix_client = Arc::new(MockPredictClient);
        let phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync> =
            Arc::new(MockRetrievalClient);
        let thunder_client = Arc::new(ThunderClient::mock());
        let strato_client = Arc::new(MockStratoClient::default());
        let tweet_mixer_client: Arc<dyn TweetMixerClient> = Arc::new(MockTweetMixerClient);
        let simclusters_ann_client: Arc<dyn SimClustersAnnClient + Send + Sync> =
            Arc::new(MockSimClustersAnnClient);
        let tes_client = Arc::new(MockTESClient::default());
        let media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync> =
            Arc::new(MockMediaInfoCacheClient::default());
        let gizmoduck_client = Arc::new(MockGizmoduckClient::default());
        let strato_vf_client = Arc::new(MockVfClient);
        let xai_vf_client = Arc::new(MockVfClient);
        let redis_client = Arc::new(MockRedisClient::default());
        let kafka_client: Arc<dyn KafkaPublisherClient> = Arc::new(MockKafkaPublisherClient);
        let reranking_kafka_client: Arc<dyn KafkaPublisherClient> =
            Arc::new(MockKafkaPublisherClient);
        let mock_socialgraph: Arc<dyn SocialGraphClientOps> = Arc::new(MockSocialGraphClient);
        let vm_ranker_client: Arc<dyn VMRankerClient> = Arc::new(MockVMRankerClient);
        let vf_safety_labels_client: Arc<dyn TweetSafetyLabelClient> =
            Arc::new(MockTweetSafetyLabelClient);
        let phoenix_request_cache_redis_atla_client = Arc::new(MockRedisClient::default());
        let phoenix_request_cache_redis_pdxa_client: Arc<dyn RedisClient + Send + Sync> =
            Arc::new(MockRedisClient::default());
        let impression_bloom_filter_client: Arc<dyn ImpressionBloomFilterClient> =
            Arc::new(MockImpressionBloomFilterClient::default());
        let ip_client = Arc::new(GeoIpLocationClient::mock());
        let user_demographics_client: Arc<dyn UserDemographicsClient> =
            Arc::new(MockUserDemographicsClient);
        let user_inferred_gender_store_client: Arc<dyn UserInferredGenderStoreClient> =
            Arc::new(MockUserInferredGenderStoreClient);
        let user_inferred_gender_grpc_client: Arc<dyn GenderPredictionGrpcClient> =
            Arc::new(MockGenderPredictionGrpcClient);
        let impressed_posts_client: Arc<dyn ImpressedPostsClient> =
            Arc::new(crate::clients::impressed_posts_client::MockImpressedPostsClient::default());
        let engagement_counts_client: Arc<dyn EngagementCountsClient> = Arc::new(
            crate::clients::engagement_counts_client::MockEngagementCountsClient::default(),
        );
        let followed_grok_topics_client: Arc<dyn FollowedGrokTopicsStoreClient> =
            Arc::new(MockFollowedGrokTopicsStoreClient);
        let followed_starter_packs_client: Arc<dyn FollowedStarterPacksStoreClient> =
            Arc::new(MockFollowedStarterPacksStoreClient);
        let user_installed_apps_client: Arc<dyn UserInstalledAppsStoreClient> =
            Arc::new(MockUserInstalledAppsStoreClient);
        let engagement_signals_client: Arc<dyn EngagementSignalsClient> =
            Arc::new(MockEngagementSignalsClient);
        let feature_switches = Arc::new(FeatureSwitches::new(vec![]).unwrap());
        PhoenixCandidatePipeline::build_with_clients(
            user_action_aggregation_client,
            phoenix_client,
            phoenix_retrieval_client,
            thunder_client,
            None,
            strato_client,
            tweet_mixer_client,
            simclusters_ann_client,
            tes_client,
            media_info_cache_client,
            gizmoduck_client,
            strato_vf_client,
            xai_vf_client,
            redis_client,
            kafka_client,
            reranking_kafka_client,
            mock_socialgraph,
            vm_ranker_client,
            vf_safety_labels_client,
            phoenix_request_cache_redis_atla_client,
            phoenix_request_cache_redis_pdxa_client,
            impression_bloom_filter_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            impressed_posts_client,
            engagement_counts_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
            user_installed_apps_client,
            engagement_signals_client,
            feature_switches,
            &super::PhoenixXdsConfig::disabled(),
            &super::VmRankerXdsConfig::disabled_with_healthz(),
            Arc::new(MockSidClient) as Arc<dyn SidClient>,
        )
        .await
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, PostCandidate> for PhoenixCandidatePipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        &self.query_hydrators
    }

    fn sources(&self) -> &[Box<dyn Source<ScoredPostsQuery, PostCandidate>>] {
        &self.sources
    }
    fn hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>] {
        &self.hydrators
    }

    fn filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        &self.filters
    }

    fn scorers(&self) -> &[Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>] {
        &self.scorers
    }

    fn selector(&self) -> &dyn Selector<ScoredPostsQuery, PostCandidate> {
        &self.selector
    }

    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>] {
        &self.post_selection_hydrators
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        &self.post_selection_filters
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        params::RESULT_SIZE
    }
}

#[cfg(test)]
mod tests {
    use crate::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
    use crate::models::query::ScoredPostsQuery;
    use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_candidate_pipeline() -> Result<(), Box<dyn std::error::Error>> {
        xai_init_utils::init().rustls();
        let pipeline = PhoenixCandidatePipeline::mock().await;
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_enable_scoring_sequence_hydration".to_string(),
            "true",
        );
        let pipeline_result = pipeline
            .execute(ScoredPostsQuery {
                user_id: 12,
                params: results.into(),
                ..Default::default()
            })
            .await;
        let hydrated_query = pipeline_result.query;
        assert_eq!(hydrated_query.user_id, 12);
        assert!(hydrated_query.scoring_sequence.is_some());
        Ok(())
    }

    #[test]
    fn test_bulk_topic_detection() {
        let query = ScoredPostsQuery {
            topic_ids: vec![1, 2, 3, 4, 5, 6],
            ..Default::default()
        };
        assert!(!query.is_bulk_topic_request());

        let query = ScoredPostsQuery {
            topic_ids: vec![1, 2, 3, 4, 5, 6, 7],
            ..Default::default()
        };
        assert!(query.is_bulk_topic_request());

        let query = ScoredPostsQuery {
            topic_ids: vec![],
            ..Default::default()
        };
        assert!(!query.is_bulk_topic_request());
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_feature_switches() {
        use std::env;
        use xai_feature_switches::{FeatureSwitches, RecipientBuilder};

        let fs_local = format!(
            "/Users/{}/workspace/config/features/home-mixer/main/for_you.yml",
            env::var("USER").unwrap()
        );
        let fs = FeatureSwitches::load_file(fs_local).expect("failed to load features.yml");
        let recipient = RecipientBuilder::new().build();
        let results = fs.match_recipient(&recipient);
        let results = results.get_i64("for_you_server_max_results");
        assert_eq!(results, Some(35));

        let fs_local = format!(
            "/Users/{}/workspace/config/features/home-mixer/main/home_mixer.yml",
            env::var("USER").unwrap()
        );
        let fs = FeatureSwitches::load_file(fs_local).expect("failed to load features.yml");
        let recipient = RecipientBuilder::new().build();
        let results = fs.match_recipient(&recipient);
        let results = results.get_bool("home_mixer_enable_new_tweets_pill_avatars");
        assert_eq!(results, Some(true));

        let fs_local = format!(
            "/Users/{}/workspace/config/features/home-mixer/main/scored_tweets.yml",
            env::var("USER").unwrap()
        );
        let fs = FeatureSwitches::load_file(fs_local).expect("failed to load features.yml");
        let recipient = RecipientBuilder::new().build();
        let results = fs.match_recipient(&recipient);
        let results = results.get_i64("scored_tweets_default_requested_max_results");
        assert_eq!(results, Some(50));

        let fs_local = format!(
            "/Users/{}/workspace/config/features/home-mixer/main/scored_video_tweets.yml",
            env::var("USER").unwrap()
        );
        let fs = FeatureSwitches::load_file(fs_local).expect("failed to load features.yml");
        let recipient = RecipientBuilder::new().build();
        let results = fs.match_recipient(&recipient);
        let results =
            results.get_string("scored_video_tweets_in_network_earlybird_tensorflow_model");
        assert_eq!(results, Some("timelines_unified_prod"));
    }
}
