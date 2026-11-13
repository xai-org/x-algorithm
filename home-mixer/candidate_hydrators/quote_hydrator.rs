use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableQuotedVqvDurationCheck;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::media_info_cache_client::MediaInfoCacheClient;
use xai_candidate_pipeline::component_library::clients::SocialGraphClientOps;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, TwoPhaseHydrator};
use xai_stats_receiver::global_stats_receiver;

const QUOTED_CONTENT_CACHE_METRIC: &str = "QuoteHydrator.quoted_content_cache";
const QUOTED_CONTENT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);

pub struct QuoteHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub socialgraph_client: Arc<dyn SocialGraphClientOps>,
    pub media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
    pub cache: QuickCache<u64, QuoteCacheValue>,
    pub quoted_content_cache: QuickCache<u64, QuotedContent>,
}

impl QuoteHydrator {
    pub async fn new(
        tes_client: Arc<dyn TESClient + Send + Sync>,
        socialgraph_client: Arc<dyn SocialGraphClientOps>,
        media_info_cache_client: Arc<dyn MediaInfoCacheClient + Send + Sync>,
    ) -> Self {
        Self {
            tes_client,
            socialgraph_client,
            media_info_cache_client,
            cache: default_quick_cache(),
            quoted_content_cache: default_quick_cache(),
        }
    }

    async fn get_quoted_video_durations(
        &self,
        quoted_tweet_ids: Vec<u64>,
    ) -> HashMap<u64, Option<i32>> {
        if quoted_tweet_ids.is_empty() {
            return HashMap::new();
        }
        let result = tokio::time::timeout(
            QUOTED_CONTENT_TIMEOUT,
            self.tes_client.get_min_video_durations(quoted_tweet_ids),
        )
        .await;
        match result {
            Ok(durations) => durations
                .into_iter()
                .filter_map(|(id, result)| result.ok().map(|d| (id, d.map(|v| v as i32))))
                .collect(),
            Err(_) => HashMap::new(),
        }
    }

    async fn get_blocked_by(&self, viewer_id: u64, quoted_user_ids: Vec<u64>) -> HashSet<u64> {
        if quoted_user_ids.is_empty() {
            return HashSet::new();
        }
        self.socialgraph_client
            .check_blocked_by(viewer_id, &quoted_user_ids)
            .await
            .unwrap_or_default()
    }

    async fn get_quoted_content(&self, quoted_tweet_ids: Vec<u64>) -> HashMap<u64, QuotedContent> {
        let mut content = HashMap::with_capacity(quoted_tweet_ids.len());
        let mut misses = Vec::new();
        for id in quoted_tweet_ids {
            match self.quoted_content_cache.get(&id).await {
                Some(cached) => {
                    content.insert(id, cached);
                }
                None => misses.push(id),
            }
        }
        stat_quoted_content_cache(content.len(), misses.len());
        if misses.is_empty() {
            return content;
        }

        let (core_data, media_info) = tokio::join!(
            tokio::time::timeout(
                QUOTED_CONTENT_TIMEOUT,
                self.tes_client.get_tweet_core_datas(misses.clone())
            ),
            tokio::time::timeout(
                QUOTED_CONTENT_TIMEOUT,
                self.media_info_cache_client.multi_get_media_info(&misses)
            ),
        );
        let core_data = core_data.unwrap_or_default();
        let media_info = media_info.unwrap_or_default();

        for id in misses {
            let text = match core_data.get(&id) {
                Some(Ok(Some(data))) => Some(data.text.clone()),
                Some(Ok(None)) => Some(String::new()),
                _ => None,
            };
            let media = match media_info.get(&id) {
                Some(Ok(Some(info))) => Some(QuotedMedia {
                    has_media: info.has_media,
                    has_photo: info.has_photo,
                    has_video: info.has_video,
                    media_count: info.media_count.clamp(0, i32::MAX as i64) as i32,
                    max_video_duration_ms: info
                        .video_durations_ms
                        .iter()
                        .copied()
                        .max()
                        .map(|v| v as i32),
                }),
                Some(Ok(None)) => Some(QuotedMedia::default()),
                _ => None,
            };
            let value = QuotedContent { text, media };
            if value.text.is_some() && value.media.is_some() {
                self.quoted_content_cache.insert(id, value.clone()).await;
            }
            content.insert(id, value);
        }
        content
    }
}

fn stat_quoted_content_cache(hits: usize, misses: usize) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    if hits > 0 {
        receiver.incr(
            QUOTED_CONTENT_CACHE_METRIC,
            &[("requests", "cache_hit")],
            hits as u64,
        );
    }
    if misses > 0 {
        receiver.incr(
            QUOTED_CONTENT_CACHE_METRIC,
            &[("requests", "cache_miss")],
            misses as u64,
        );
    }
}

#[derive(Clone, Debug)]
pub struct QuoteCacheValue {
    pub quoted_tweet_id: Option<u64>,
    pub quoted_user_id: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuotedMedia {
    pub has_media: bool,
    pub has_photo: bool,
    pub has_video: bool,
    pub media_count: i32,
    pub max_video_duration_ms: Option<i32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuotedContent {
    pub text: Option<String>,
    pub media: Option<QuotedMedia>,
}

pub struct Quote {
    tweet_id: Option<u64>,
    user_id: Option<u64>,
    author_blocks_viewer: bool,
    video_duration_ms: Option<i32>,
}

pub struct Quotes {
    quotes: Vec<Quote>,
    content: HashMap<u64, QuotedContent>,
}

#[async_trait]
impl TwoPhaseHydrator<ScoredPostsQuery, PostCandidate> for QuoteHydrator {
    type Fetched = Quotes;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        !query.has_cached_posts
    }

    async fn fetch(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Quotes, String> {
        let tweet_ids: Vec<u64> = candidates.iter().map(|c| c.tweet_id).collect();

        let mut cache_misses: Vec<u64> = Vec::new();
        let mut resolved: Vec<(u64, Option<u64>, Option<u64>)> =
            Vec::with_capacity(tweet_ids.len());

        for &tweet_id in &tweet_ids {
            if let Some(cached) = self.cache.get(&tweet_id).await {
                resolved.push((tweet_id, cached.quoted_tweet_id, cached.quoted_user_id));
            } else {
                cache_misses.push(tweet_id);
                resolved.push((tweet_id, None, None));
            }
        }

        if !cache_misses.is_empty() {
            let quoted_tweets = self
                .tes_client
                .get_quoted_tweets(cache_misses.clone())
                .await;

            for entry in resolved.iter_mut() {
                let tweet_id = entry.0;
                if !cache_misses.contains(&tweet_id) {
                    continue;
                }
                let (qt_tweet_id, qt_user_id) = match quoted_tweets.get(&tweet_id) {
                    Some(Ok(Some(qt))) => (Some(qt.tweet_id), Some(qt.user_id)),
                    _ => (None, None),
                };
                entry.1 = qt_tweet_id;
                entry.2 = qt_user_id;

                self.cache
                    .insert(
                        tweet_id,
                        QuoteCacheValue {
                            quoted_tweet_id: qt_tweet_id,
                            quoted_user_id: qt_user_id,
                        },
                    )
                    .await;
            }
        }

        let quoted_user_ids: Vec<u64> = resolved
            .iter()
            .filter_map(|(_, _, uid)| *uid)
            .collect::<HashSet<u64>>()
            .into_iter()
            .collect();

        let unique_quoted_tweet_ids: Vec<u64> = resolved
            .iter()
            .filter_map(|(_, qt_id, _)| *qt_id)
            .collect::<HashSet<u64>>()
            .into_iter()
            .collect();

        let fetch_quoted_duration = query.params.get(EnableQuotedVqvDurationCheck);
        let quoted_tweet_ids: Vec<u64> = if fetch_quoted_duration {
            unique_quoted_tweet_ids.clone()
        } else {
            Vec::new()
        };

        let (blocked_by, quoted_durations, quoted_content) = tokio::join!(
            self.get_blocked_by(query.user_id, quoted_user_ids),
            self.get_quoted_video_durations(quoted_tweet_ids),
            self.get_quoted_content(unique_quoted_tweet_ids),
        );

        let quotes = resolved
            .iter()
            .map(|&(_, tweet_id, user_id)| Quote {
                tweet_id,
                user_id,
                author_blocks_viewer: user_id.is_some_and(|uid| blocked_by.contains(&uid)),
                video_duration_ms: tweet_id
                    .and_then(|id| quoted_durations.get(&id).copied())
                    .flatten(),
            })
            .collect();
        Ok(Quotes {
            quotes,
            content: quoted_content,
        })
    }

    fn apply(&self, _query: &ScoredPostsQuery, candidates: &mut [PostCandidate], fetched: Quotes) {
        for (candidate, quote) in candidates.iter_mut().zip(fetched.quotes) {
            let content = quote.tweet_id.and_then(|id| fetched.content.get(&id));
            let media = content.and_then(|content| content.media.as_ref());
            candidate.quoted_tweet_id = quote.tweet_id;
            candidate.quoted_user_id = quote.user_id;
            candidate.quoted_author_blocks_viewer = Some(quote.author_blocks_viewer);
            candidate.quoted_video_duration_ms = quote.video_duration_ms;
            candidate.quoted_tweet_text = content.and_then(|content| content.text.clone());
            candidate.quoted_has_media = media.map(|media| media.has_media);
            candidate.quoted_has_photo = media.map(|media| media.has_photo);
            candidate.quoted_has_video = media.map(|media| media.has_video);
            candidate.quoted_media_count = media.map(|media| media.media_count);
            candidate.quoted_max_video_duration_ms =
                media.and_then(|media| media.max_video_duration_ms);
        }
    }

    fn merge(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.quoted_tweet_id = hydrated.quoted_tweet_id;
        candidate.quoted_user_id = hydrated.quoted_user_id;
        candidate.quoted_author_blocks_viewer = hydrated.quoted_author_blocks_viewer;
        candidate.quoted_video_duration_ms = hydrated.quoted_video_duration_ms;
        candidate.quoted_tweet_text = hydrated.quoted_tweet_text;
        candidate.quoted_has_media = hydrated.quoted_has_media;
        candidate.quoted_has_photo = hydrated.quoted_has_photo;
        candidate.quoted_has_video = hydrated.quoted_has_video;
        candidate.quoted_media_count = hydrated.quoted_media_count;
        candidate.quoted_max_video_duration_ms = hydrated.quoted_max_video_duration_ms;
    }
}
