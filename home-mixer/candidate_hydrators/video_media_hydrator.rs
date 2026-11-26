use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::util::video_carousel;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};
use xai_core_entities::entities::MediaEntity;
use xai_x_thrift::tweet_media::MediaInfo;

#[derive(Clone, Debug, Default)]
pub struct VideoMedia {
    aspect_ratio: Option<f32>,
    media_id: Option<u64>,
    md5_hashes: Option<Vec<String>>,
}

pub struct VideoMediaHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: QuickCache<u64, VideoMedia>,
}

impl VideoMediaHydrator {
    pub fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        Self {
            tes_client,
            cache: default_quick_cache(),
        }
    }
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for VideoMediaHydrator {
    type Fetched = Vec<Option<VideoMedia>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<VideoMedia>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<VideoMedia>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<VideoMedia>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.video_aspect_ratio = value.aspect_ratio;
                candidate.video_media_id = value.media_id;
                candidate.video_md5_hashes = value.md5_hashes;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.video_aspect_ratio = hydrated.video_aspect_ratio;
        candidate.video_media_id = hydrated.video_media_id;
        candidate.video_md5_hashes = hydrated.video_md5_hashes;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for VideoMediaHydrator {
    type CacheKey = u64;
    type Value = VideoMedia;

    fn cache_store(&self) -> &dyn CacheStore<u64, VideoMedia> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.get_original_tweet_id()
    }

    fn existing_value(&self, candidate: &PostCandidate) -> Option<VideoMedia> {
        (candidate.has_video != Some(true) || candidate.video_aspect_ratio.is_some()).then(|| {
            VideoMedia {
                aspect_ratio: candidate.video_aspect_ratio,
                media_id: candidate.video_media_id,
                md5_hashes: candidate.video_md5_hashes.clone(),
            }
        })
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<VideoMedia>> {
        let tweet_ids: Vec<u64> = misses.iter().map(|c| c.get_original_tweet_id()).collect();

        let media_entities = self
            .tes_client
            .get_tweet_media_entities(tweet_ids.clone())
            .await;

        tweet_ids
            .iter()
            .map(|tweet_id| match media_entities.get(tweet_id) {
                Some(Ok(entities)) => Some(video_media(
                    entities.as_ref().and_then(|entities| entities.first()),
                )),
                Some(Err(_)) => None,
                None => Some(VideoMedia::default()),
            })
            .collect()
    }
}

fn video_media(entity: Option<&MediaEntity>) -> VideoMedia {
    VideoMedia {
        aspect_ratio: entity.and_then(video_aspect_ratio),
        media_id: entity
            .and_then(|entity| entity.media_id)
            .map(|id| id as u64),
        md5_hashes: entity.and_then(md5_hashes),
    }
}

fn md5_hashes(entity: &MediaEntity) -> Option<Vec<String>> {
    let hashes = entity.additional_metadata.as_ref()?.md5_hashes.as_ref()?;
    Some(hashes.iter().filter_map(|hash| hash.hash.clone()).collect())
}

fn video_aspect_ratio(entity: &MediaEntity) -> Option<f32> {
    let Some(MediaInfo::VideoInfo(info)) = &entity.media_info else {
        return None;
    };
    let ratio = info.aspect_ratio.as_ref()?;
    let aspect_ratio = f32::from(ratio.numerator?) / f32::from(ratio.denominator?);
    (aspect_ratio.is_finite() && aspect_ratio > 0.0).then_some(aspect_ratio)
}
