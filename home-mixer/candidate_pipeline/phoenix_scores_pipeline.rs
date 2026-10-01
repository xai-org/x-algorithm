use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::candidate_hydrators::gizmoduck_hydrator::GizmoduckCandidateHydrator;
use crate::candidate_hydrators::in_network_candidate_hydrator::InNetworkCandidateHydrator;
use crate::candidate_hydrators::language_code_hydrator::LanguageCodeHydrator;
use crate::candidate_hydrators::media_info_hydrator::MediaInfoHydrator;
use crate::candidate_hydrators::quote_hydrator::QuoteHydrator;
use crate::candidate_hydrators::semantic_id_hydrator::SemanticIdHydrator;
use crate::clients::gizmoduck_client::{GizmoduckClient, MockGizmoduckClient, ProdGizmoduckClient};
use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::clients::user_action_aggregation_client::{
    MockUserActionAggregationClient, ProdUserActionAggregationClient, UserActionAggregationClient,
};
use crate::filters::core_data_hydration_filter::CoreDataHydrationFilter;
use crate::filters::result_size_filter::ResultSizeFilter;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::query_hydrators::followed_grok_topics_query_hydrator::FollowedGrokTopicsQueryHydrator;
use crate::query_hydrators::followed_starter_packs_query_hydrator::FollowedStarterPacksQueryHydrator;
use crate::query_hydrators::followed_user_ids_query_hydrator::FollowedUserIdsQueryHydrator;
use crate::query_hydrators::ip_query_hydrator::IpQueryHydrator;
use crate::query_hydrators::scoring_sequence_query_hydrator::ScoringSequenceQueryHydrator;
use crate::query_hydrators::user_demographics_query_hydrator::UserDemographicsQueryHydrator;
use crate::query_hydrators::user_inferred_gender_query_hydrator::UserInferredGenderQueryHydrator;
use crate::query_hydrators::user_installed_apps_query_hydrator::UserInstalledAppsQueryHydrator;
use crate::scorers::phoenix_scorer::PhoenixScorer;
use crate::scorers::phoenix_scores_ranking_scorer::PhoenixScoresRankingScorer;
use crate::selectors::PassthroughSelector;
use crate::side_effects::scored_stats_side_effect::ScoredStatsSideEffect;
use crate::sources::seed_candidates_source::SeedCandidatesSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
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
use xai_candidate_pipeline::component_library::clients::media_info_cache_client::{
    MediaInfoCacheClient, MockMediaInfoCacheClient, ProdMediaInfoCacheClient,
};
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    MockPredictClient, PhoenixPredictionClient, ProdPhoenixPredictionClient,
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
    MockSocialGraphClient, SocialGraphClient, SocialGraphClientOps,
};
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_geo_ip::GeoIpLocationClient;
use xai_x_rpc::wily_lookup_service::ShardCoordinate;

use super::build_phoenix_xds_client;

const RESULT_SIZE: usize = 2800;

pub struct PhoenixScoresPipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>,
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>>,
    selector: PassthroughSelector,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>>,
}

impl PhoenixScoresPipeline {
    pub(crate) async fn build_with_clients(
        user_action_aggregation_client: Arc<dyn UserActionAggregationClient + Send + Sync>,
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        xds_client: Option<Arc<dyn PhoenixPredictionClient + Send + Sync>>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
        gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
        socialgraph_client: Arc<dyn SocialGraphClientOps>,
        ip_client: Arc<GeoIpLocationClient>,
        user_demographics_client: Arc<dyn UserDemographicsClient>,
        user_inferred_gender_store_client: Arc<dyn UserInferredGenderStoreClient>,
        user_inferred_gender_grpc_client: Arc<dyn GenderPredictionGrpcClient>,
        followed_grok_topics_client: Arc<dyn FollowedGrokTopicsStoreClient>,
        followed_starter_packs_client: Arc<dyn FollowedStarterPacksStoreClient>,
        user_installed_apps_client: Arc<dyn UserInstalledAppsStoreClient>,
        sid_client: Arc<dyn SidClient>,
    ) -> PhoenixScoresPipeline {
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            Box::new(ScoringSequenceQueryHydrator::new(
                user_action_aggregation_client,
            )),
            Box::new(FollowedUserIdsQueryHydrator {
                socialgraph_client: socialgraph_client.clone(),
            }),
            Box::new(UserDemographicsQueryHydrator {
                client: user_demographics_client,
            }),
            Box::new(UserInferredGenderQueryHydrator::new(
                user_inferred_gender_store_client,
                user_inferred_gender_grpc_client,
            )),
            Box::new(FollowedGrokTopicsQueryHydrator::new(
                followed_grok_topics_client,
            )),
            Box::new(FollowedStarterPacksQueryHydrator::new(
                followed_starter_packs_client,
            )),
            Box::new(UserInstalledAppsQueryHydrator::new(
                user_installed_apps_client,
            )),
            Box::new(IpQueryHydrator { client: ip_client }),
        ];

        let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> =
            vec![Box::new(SeedCandidatesSource)];

        // TES author_id before InNetwork — do not invert. See phoenix_candidate_pipeline.
        let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(CoreDataCandidateHydrator::new(tes_client.clone()).await),
            Box::new(InNetworkCandidateHydrator),
            Box::new(GizmoduckCandidateHydrator::new(gizmoduck_client).await),
            Box::new(MediaInfoHydrator::new(media_info_cache_client.clone()).await),
            Box::new(LanguageCodeHydrator::new(tes_client.clone()).await),
            Box::new(
                QuoteHydrator::new(tes_client, socialgraph_client, media_info_cache_client).await,
            ),
            Box::new(SemanticIdHydrator::new(sid_client)),
        ];

        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> =
            vec![Box::new(CoreDataHydrationFilter)];

        let scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(PhoenixScorer {
                dispatch: {
                    let mut paths = Vec::new();
                    if let Some(xds) = xds_client {
                        paths.push(crate::util::egress::PredictionPath {
                            name: "xDS",
                            gate: crate::util::xds::XDS_GATE,
                            client: xds,
                            config: crate::util::egress::EgressConfig::DEFAULT,
                        });
                    }
                    crate::util::egress::PredictionDispatch {
                        prod: phoenix_client,
                        paths,
                        max_retries_key: "rust_home_mixer_phoenix_xds_max_retries",
                        enable_fallback_key: "rust_home_mixer_phoenix_enable_fallback",
                    }
                },
            }),
            Box::new(PhoenixScoresRankingScorer),
        ];

        let side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> =
            Arc::new(vec![Box::new(ScoredStatsSideEffect)]);

        PhoenixScoresPipeline {
            query_hydrators,
            sources,
            hydrators,
            filters,
            scorers,
            selector: PassthroughSelector,
            post_selection_filters: vec![Box::new(ResultSizeFilter)],
            side_effects,
        }
    }

    pub async fn prod(
        shard_coordinate: Option<ShardCoordinate>,
        datacenter: &str,
        phoenix_xds: &super::PhoenixXdsConfig,
    ) -> PhoenixScoresPipeline {
        let xds_client = build_phoenix_xds_client(phoenix_xds).await;

        let (
            socialgraph_client,
            user_action_aggregation_client,
            phoenix_client,
            tes_client,
            media_info_cache_client,
            gizmoduck_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
            user_installed_apps_client,
            sid_client,
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
            async { Arc::new(ProdSidClient::new()) as Arc<dyn SidClient> },
        );

        PhoenixScoresPipeline::build_with_clients(
            user_action_aggregation_client,
            phoenix_client,
            xds_client,
            tes_client,
            media_info_cache_client,
            gizmoduck_client,
            socialgraph_client,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
            user_installed_apps_client,
            sid_client,
        )
        .await
    }

    pub async fn mock() -> PhoenixScoresPipeline {
        let user_action_aggregation_client = Arc::new(MockUserActionAggregationClient);
        let phoenix_client = Arc::new(MockPredictClient);
        let tes_client = Arc::new(MockTESClient::default());
        let media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync> =
            Arc::new(MockMediaInfoCacheClient::default());
        let gizmoduck_client = Arc::new(MockGizmoduckClient::default());
        let mock_socialgraph: Arc<dyn SocialGraphClientOps> = Arc::new(MockSocialGraphClient);
        let ip_client = Arc::new(GeoIpLocationClient::mock());
        let user_demographics_client: Arc<dyn UserDemographicsClient> =
            Arc::new(MockUserDemographicsClient);
        let user_inferred_gender_store_client: Arc<dyn UserInferredGenderStoreClient> =
            Arc::new(MockUserInferredGenderStoreClient);
        let user_inferred_gender_grpc_client: Arc<dyn GenderPredictionGrpcClient> =
            Arc::new(MockGenderPredictionGrpcClient);
        let followed_grok_topics_client: Arc<dyn FollowedGrokTopicsStoreClient> =
            Arc::new(MockFollowedGrokTopicsStoreClient);
        let followed_starter_packs_client: Arc<dyn FollowedStarterPacksStoreClient> =
            Arc::new(MockFollowedStarterPacksStoreClient);
        let user_installed_apps_client: Arc<dyn UserInstalledAppsStoreClient> =
            Arc::new(MockUserInstalledAppsStoreClient);
        PhoenixScoresPipeline::build_with_clients(
            user_action_aggregation_client,
            phoenix_client,
            None,
            tes_client,
            media_info_cache_client,
            gizmoduck_client,
            mock_socialgraph,
            ip_client,
            user_demographics_client,
            user_inferred_gender_store_client,
            user_inferred_gender_grpc_client,
            followed_grok_topics_client,
            followed_starter_packs_client,
            user_installed_apps_client,
            Arc::new(MockSidClient) as Arc<dyn SidClient>,
        )
        .await
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, PostCandidate> for PhoenixScoresPipeline {
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
        &[]
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        &self.post_selection_filters
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        RESULT_SIZE
    }
}
