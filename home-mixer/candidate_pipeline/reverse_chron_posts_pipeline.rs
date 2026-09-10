use crate::candidate_hydrators::ads_brand_safety_vf_hydrator::AdsBrandSafetyVfHydrator;
use crate::candidate_hydrators::conversation_gap_ancestor_hydrator::ConversationGapAncestorHydrator;
use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
use crate::candidate_hydrators::following_blocked_by_hydrator::FollowingBlockedByHydrator;
use crate::candidate_hydrators::media_info_hydrator::MediaInfoHydrator;
use crate::candidate_hydrators::quoted_post_text_hydrator::QuotedPostTextHydrator;
use crate::candidate_hydrators::tweet_type_metrics_hydrator::TweetTypeMetricsHydrator;
use crate::candidate_hydrators::vf_following_candidate_hydrator::VFFollowingCandidateHydrator;
use crate::clients::night_owl_client::{MockNightOwlClient, NightOwlClient, ProdNightOwlClient};
use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use crate::clients::tweet_entity_service_client::{MockTESClient, ProdTESClient, TESClient};
use crate::filters::ancillary_vf_filter::AncillaryVFFilter;
use crate::filters::author_socialgraph_filter::AuthorSocialgraphFilter;
use crate::filters::following_retweet_deduplication_filter::FollowingRetweetDeduplicationFilter;
use crate::filters::following_viewer_muted_keyword_filter::FollowingViewerMutedKeywordFilter;
use crate::filters::self_reply_chain_filter::SelfReplyChainFilter;
use crate::filters::vf_filter::VFFilter;
use crate::filters::video_filter::VideoFilter;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::FOLLOWING_POST_FETCH_SIZE;
use crate::selectors::PassthroughSelector;
use crate::sources::following_night_owl_source::FollowingNightOwlSource;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_candidate_pipeline::component_library::clients::media_info_cache_client::{
    MediaInfoCacheClient, MockMediaInfoCacheClient, ProdMediaInfoCacheClient,
};
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
            socialgraph_client,
            media_info_cache_client,
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
                    ProdMediaInfoCacheClient::new(datacenter, "home-mixer")
                        .await
                        .expect("Failed to create MediaInfoCacheClient"),
                ) as Arc<dyn MediaInfoCacheClient + Send + Sync>
            },
        );

        Self::build(
            night_owl_client,
            tes_client,
            strato_vf_client,
            xai_vf_client,
            vf_safety_labels_client,
            socialgraph_client,
            media_info_cache_client,
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
            Arc::new(MockSocialGraphClient) as Arc<dyn SocialGraphClientOps>,
            Arc::new(MockMediaInfoCacheClient::default())
                as Arc<dyn MediaInfoCacheClient + Send + Sync>,
        )
        .await
    }

    async fn build(
        night_owl_client: Arc<dyn NightOwlClient>,
        tes_client: Arc<dyn TESClient + Send + Sync>,
        strato_vf_client: Arc<dyn VfClient + Send + Sync>,
        xai_vf_client: Arc<dyn VfClient + Send + Sync>,
        vf_safety_labels_client: Arc<dyn TweetSafetyLabelClient>,
        socialgraph_client: Arc<dyn SocialGraphClientOps>,
        media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
    ) -> Self {
        let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> =
            vec![Box::new(FollowingNightOwlSource {
                client: night_owl_client,
            })];

        let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(CoreDataCandidateHydrator::new(Arc::clone(&tes_client)).await),
            Box::new(ConversationGapAncestorHydrator::new(Arc::clone(
                &tes_client,
            ))),
            Box::new(QuotedPostTextHydrator::new(tes_client)),
            Box::new(MediaInfoHydrator::new(media_info_cache_client).await),
        ];

        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(FollowingRetweetDeduplicationFilter),
            Box::new(FollowingViewerMutedKeywordFilter::new()),
            Box::new(SelfReplyChainFilter),
            Box::new(VideoFilter),
        ];

        let post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(FollowingBlockedByHydrator::new(socialgraph_client).await),
            Box::new(VFFollowingCandidateHydrator::new(
                strato_vf_client,
                xai_vf_client,
            )),
            Box::new(AdsBrandSafetyVfHydrator {
                client: vf_safety_labels_client,
            }),
            Box::new(TweetTypeMetricsHydrator::new()),
        ];

        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            Box::new(AuthorSocialgraphFilter),
            Box::new(VFFilter),
            Box::new(AncillaryVFFilter),
        ];

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xai_x_thrift::tweet_media_info::TweetMediaInfo;

    fn video_info(video_durations_ms: Vec<i64>) -> TweetMediaInfo {
        TweetMediaInfo::new(true, true, true, 1, video_durations_ms)
    }

    #[tokio::test]
    async fn mock_pipeline_wires_media_info_and_video_filter() {
        let pipeline = ReverseChronPostsPipeline::mock().await;
        assert_eq!(pipeline.hydrators().len(), 4);
        assert_eq!(pipeline.filters().len(), 4);
    }

    #[tokio::test]
    async fn exclude_videos_drops_native_video_after_media_info() {
        let mut entries = HashMap::new();
        entries.insert(1u64, Some(video_info(vec![5_000])));
        let hydrator = MediaInfoHydrator::new(Arc::new(MockMediaInfoCacheClient {
            media_info: entries,
        }))
        .await;

        let query = ScoredPostsQuery {
            exclude_videos: true,
            ..Default::default()
        };
        let mut candidates = vec![
            PostCandidate {
                tweet_id: 1,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                ..Default::default()
            },
        ];

        let hydrated = hydrator.hydrate(&query, &candidates).await;
        hydrator.update(&mut candidates[0], hydrated[0].clone().unwrap());
        hydrator.update(&mut candidates[1], hydrated[1].clone().unwrap());
        assert_eq!(candidates[0].min_video_duration_ms, Some(5_000));
        assert_eq!(candidates[1].min_video_duration_ms, None);

        let result = VideoFilter.filter(&query, candidates);
        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 2);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 1);
    }

    #[tokio::test]
    async fn keeps_videos_when_exclude_videos_is_off() {
        let query = ScoredPostsQuery::default();
        assert!(!VideoFilter.enable(&query));
    }
}
