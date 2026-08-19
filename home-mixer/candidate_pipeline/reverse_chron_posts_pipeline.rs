use crate::candidate_hydrators::ads_brand_safety_vf_hydrator::AdsBrandSafetyVfHydrator;
use crate::candidate_hydrators::conversation_gap_ancestor_hydrator::ConversationGapAncestorHydrator;
use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::candidate_hydrators::tweet_type_metrics_hydrator::TweetTypeMetricsHydrator;
use crate::candidate_hydrators::vf_following_candidate_hydrator::VFFollowingCandidateHydrator;
use crate::clients::night_owl_client::{MockNightOwlClient, NightOwlClient, ProdNightOwlClient};
use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::filters::ancillary_vf_filter::AncillaryVFFilter;
use crate::filters::following_retweet_deduplication_filter::FollowingRetweetDeduplicationFilter;
use crate::filters::self_reply_chain_filter::SelfReplyChainFilter;
use crate::filters::vf_filter::VFFilter;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::FOLLOWING_POST_FETCH_SIZE;
use crate::selectors::PassthroughSelector;
use crate::sources::following_night_owl_source::FollowingNightOwlSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::filter::Filter;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
use xai_candidate_pipeline::scorer::Scorer;
use xai_candidate_pipeline::selector::Selector;
use xai_candidate_pipeline::side_effect::SideEffect;
use xai_candidate_pipeline::source::Source;
use xai_visibility_filtering::tweet_safety_label::{
    MockTweetSafetyLabelClient, ProdTweetSafetyLabelClient, TweetSafetyLabelClient,
};
use xai_visibility_filtering::vf_client::{MockVfClient, StratoVfClient, VfClient, XaiVfClient};

pub struct ReverseChronPostsPipeline {
    sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>,
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    selector: PassthroughSelector,
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>>,
}

impl ReverseChronPostsPipeline {
    pub async fn new(datacenter: &str) -> Self {
        let (
            night_owl_client,
            tes_client,
            strato_vf_client,
            xai_vf_client,
            vf_safety_labels_client,
        ) = tokio::join!(
            async {
                Arc::new(
                    ProdNightOwlClient::new(datacenter)
                        .await
                        .expect("Failed to create NightOwl client"),
                ) as Arc<dyn NightOwlClient>
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
                    ProdTweetSafetyLabelClient::new(datacenter)
                        .await
                        .expect("Failed to create VF SafetyLabels client")
                        .with_timeout_ms(500)
                        .with_max_batch_size(50),
                ) as Arc<dyn TweetSafetyLabelClient>
            },
        );

        Self::build(
            night_owl_client,
            tes_client,
            strato_vf_client,
            xai_vf_client,
            vf_safety_labels_client,
        )
        .await
    }

    pub async fn mock() -> Self {
        Self::build(
            Arc::new(MockNightOwlClient) as Arc<dyn NightOwlClient>,
            Arc::new(MockTESClient::default()) as Arc<dyn TESClient + Send + Sync>,
            Arc::new(MockVfClient) as Arc<dyn VfClient + Send + Sync>,
            Arc::new(MockVfClient) as Arc<dyn VfClient + Send + Sync>,
            Arc::new(MockTweetSafetyLabelClient) as Arc<dyn TweetSafetyLabelClient>,
        )
        .await
    }

    async fn build(
        night_owl_client: Arc<dyn NightOwlClient>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        strato_vf_client: Arc<dyn VfClient + Send + Sync>,
        xai_vf_client: Arc<dyn VfClient + Send + Sync>,
        vf_safety_labels_client: Arc<dyn TweetSafetyLabelClient>,
    ) -> Self {
        let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> =
            vec![Box::new(FollowingNightOwlSource {
                client: night_owl_client,
            })];

        let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(CoreDataCandidateHydrator::new(Arc::clone(&tes_client)).await),
            Box::new(ConversationGapAncestorHydrator::new(tes_client)),
        ];

        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(FollowingRetweetDeduplicationFilter),
            Box::new(SelfReplyChainFilter),
        ];

        let post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(VFFollowingCandidateHydrator::new(
                strato_vf_client,
                xai_vf_client,
            )),
            Box::new(AdsBrandSafetyVfHydrator {
                client: vf_safety_labels_client,
            }),
            Box::new(TweetTypeMetricsHydrator::new()),
        ];

        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> =
            vec![Box::new(VFFilter), Box::new(AncillaryVFFilter)];

        Self {
            sources,
            hydrators,
            filters,
            post_selection_hydrators,
            post_selection_filters,
            selector: PassthroughSelector,
            side_effects: Arc::new(vec![]),
        }
    }
}

#[async_trait]
impl CandidatePipeline<ScoredPostsQuery, PostCandidate> for ReverseChronPostsPipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        &[]
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
        &[]
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
        FOLLOWING_POST_FETCH_SIZE
    }
}
