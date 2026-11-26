use std::sync::Arc;

use tonic::async_trait;
use xai_candidate_pipeline::candidate_pipeline::PipelineStage;
use xai_candidate_pipeline::component_library::clients::media_info_cache_client::MediaInfoCacheClient;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedFetch, TwoPhaseHydrator};

use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaInfoCacheValue {
    has_media: Option<bool>,
    has_photo: Option<bool>,
    has_video: Option<bool>,
    media_count: Option<i32>,
    min_video_duration_ms: Option<i32>,
    max_video_duration_ms: Option<i32>,
}

impl MediaInfoCacheValue {
    fn no_media() -> Self {
        Self {
            has_media: Some(false),
            has_photo: Some(false),
            has_video: Some(false),
            media_count: Some(0),
            min_video_duration_ms: None,
            max_video_duration_ms: None,
        }
    }
}

pub struct MediaInfoHydrator {
    pub media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
    pub cache: QuickCache<u64, MediaInfoCacheValue>,
}

impl MediaInfoHydrator {
    pub async fn new(media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>) -> Self {
        let cache = default_quick_cache();
        Self {
            media_info_cache_client,
            cache,
        }
    }
}

fn min_video_duration_ms(durations: &[i64]) -> Option<i32> {
    durations.iter().copied().min().map(|v| v as i32)
}

fn max_video_duration_ms(durations: &[i64]) -> Option<i32> {
    durations.iter().copied().max().map(|v| v as i32)
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for MediaInfoHydrator {
    type Fetched = Vec<Option<MediaInfoCacheValue>>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<Option<MediaInfoCacheValue>>, String> {
        self.fetch_cached(query, candidates, PipelineStage::Hydrator)
            .await
    }

    async fn fetch_for_stage(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        stage: PipelineStage,
    ) -> Result<Vec<Option<MediaInfoCacheValue>>, String> {
        self.fetch_cached(query, candidates, stage).await
    }

    fn apply(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &mut [PostCandidate],
        values: Vec<Option<MediaInfoCacheValue>>,
    ) {
        for (candidate, value) in candidates.iter_mut().zip(values) {
            if let Some(value) = value {
                candidate.has_media = value.has_media;
                candidate.has_photo = value.has_photo;
                candidate.has_video = value.has_video;
                candidate.media_count = value.media_count;
                candidate.min_video_duration_ms = value.min_video_duration_ms;
                candidate.max_video_duration_ms = value.max_video_duration_ms;
            }
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.has_media = hydrated.has_media;
        candidate.has_photo = hydrated.has_photo;
        candidate.has_video = hydrated.has_video;
        candidate.media_count = hydrated.media_count;
        candidate.min_video_duration_ms = hydrated.min_video_duration_ms;
        candidate.max_video_duration_ms = hydrated.max_video_duration_ms;
    }
}

#[async_trait]
impl CachedFetch<ScoredPostsQuery, PostCandidate> for MediaInfoHydrator {
    type CacheKey = u64;
    type Value = MediaInfoCacheValue;

    fn cache_store(&self) -> &dyn CacheStore<u64, MediaInfoCacheValue> {
        &self.cache
    }

    fn cache_key(&self, _query: &ScoredPostsQuery, candidate: &PostCandidate) -> u64 {
        candidate.get_original_tweet_id()
    }

    async fn fetch_values(
        &self,
        _query: &ScoredPostsQuery,
        misses: &[&PostCandidate],
    ) -> Vec<Option<MediaInfoCacheValue>> {
        let tweet_ids: Vec<u64> = misses.iter().map(|c| c.get_original_tweet_id()).collect();

        let media_info = self
            .media_info_cache_client
            .multi_get_media_info(&tweet_ids)
            .await;

        tweet_ids
            .iter()
            .map(|tweet_id| match media_info.get(tweet_id) {
                Some(Ok(Some(info))) => Some(MediaInfoCacheValue {
                    has_media: Some(info.has_media),
                    has_photo: Some(info.has_photo),
                    has_video: Some(info.has_video),
                    media_count: Some(info.media_count.clamp(0, i32::MAX as i64) as i32),
                    min_video_duration_ms: min_video_duration_ms(&info.video_durations_ms),
                    max_video_duration_ms: max_video_duration_ms(&info.video_durations_ms),
                }),
                Some(Ok(None)) | None => Some(MediaInfoCacheValue::no_media()),
                Some(Err(_)) => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use xai_candidate_pipeline::component_library::clients::media_info_cache_client::MockMediaInfoCacheClient;
    use xai_candidate_pipeline::hydrator::Hydrator;
    use xai_x_thrift::tweet_media_info::TweetMediaInfo;

    fn mock_client(entries: HashMap<u64, Option<TweetMediaInfo>>) -> Arc<dyn MediaInfoCacheClient> {
        Arc::new(MockMediaInfoCacheClient {
            media_info: entries,
        })
    }

    fn info(has_media: bool, video_durations_ms: Vec<i64>) -> TweetMediaInfo {
        TweetMediaInfo::new(
            has_media,
            has_media,
            !video_durations_ms.is_empty(),
            if has_media { 1 } else { 0 },
            video_durations_ms,
        )
    }

    #[tokio::test]
    async fn test_has_media_and_duration() {
        let mut entries = HashMap::new();
        entries.insert(1001u64, Some(info(true, vec![5000, 3000])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1001,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        let c = result[0].as_ref().unwrap();
        assert_eq!(c.has_media, Some(true));
        assert_eq!(c.has_photo, Some(true));
        assert_eq!(c.has_video, Some(true));
        assert_eq!(c.media_count, Some(1));
        assert_eq!(c.min_video_duration_ms, Some(3000));
        assert_eq!(c.max_video_duration_ms, Some(5000));
    }

    #[tokio::test]
    async fn test_has_media_false_no_duration() {
        let mut entries = HashMap::new();
        entries.insert(1002u64, Some(info(false, vec![])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1002,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        let c = result[0].as_ref().unwrap();
        assert_eq!(c.has_media, Some(false));
        assert_eq!(c.has_video, Some(false));
        assert_eq!(c.media_count, Some(0));
        assert_eq!(c.min_video_duration_ms, None);
        assert_eq!(c.max_video_duration_ms, None);
    }

    #[tokio::test]
    async fn test_miss_is_no_media() {
        let hydrator = MediaInfoHydrator::new(mock_client(HashMap::new())).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1003,
            ..Default::default()
        }];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
        let c = result[0].as_ref().unwrap();
        assert_eq!(c.has_media, Some(false));
        assert_eq!(c.min_video_duration_ms, None);
    }

    #[tokio::test]
    async fn test_caching() {
        let mut entries = HashMap::new();
        entries.insert(1004u64, Some(info(true, vec![1200])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![PostCandidate {
            tweet_id: 1004,
            ..Default::default()
        }];

        let first = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert_eq!(first[0].as_ref().unwrap().has_media, Some(true));
        assert_eq!(first[0].as_ref().unwrap().min_video_duration_ms, Some(1200));

        let second = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;
        assert_eq!(second[0].as_ref().unwrap().has_media, Some(true));
        assert_eq!(
            second[0].as_ref().unwrap().min_video_duration_ms,
            Some(1200)
        );
    }

    #[tokio::test]
    async fn test_multiple_candidates() {
        let mut entries = HashMap::new();
        entries.insert(2001u64, Some(info(true, vec![1000, 2000])));
        entries.insert(2002u64, Some(info(false, vec![])));

        let hydrator = MediaInfoHydrator::new(mock_client(entries)).await;

        let candidates = vec![
            PostCandidate {
                tweet_id: 2001,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2002,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2003,
                ..Default::default()
            },
        ];

        let result = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].as_ref().unwrap().has_media, Some(true));
        assert_eq!(
            result[0].as_ref().unwrap().min_video_duration_ms,
            Some(1000)
        );
        assert_eq!(result[1].as_ref().unwrap().has_media, Some(false));
        assert_eq!(result[1].as_ref().unwrap().min_video_duration_ms, None);
        assert_eq!(result[2].as_ref().unwrap().has_media, Some(false));
        assert_eq!(result[2].as_ref().unwrap().min_video_duration_ms, None);
    }
}
