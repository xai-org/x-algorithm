use crate::hydration::batch::TweetHydrationBatch;
use crate::hydration::fallback_cache::{FallbackCache, FallbackCacheMode};
use crate::hydration::metrics::{record_batch_size, timed_keyed_rpc, timed_results};
use crate::models::{
    CoreFeature, MediaFeature, NsfwFeature, TakedownFeature, TweetCandidateInput, TweetFeatures,
    TweetId,
};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use xai_core_entities::entities::{EditControl, MediaEntities, PureCoreData, TakedownReason};
use xai_core_entities::tweet_entity_service_client::TESClient;

const CLIENT_TIMEOUT: Duration = Duration::from_millis(150);
const CLIENT: &str = "tes";
const CACHE_CAPACITY: usize = 1_000_000;

pub struct TesHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    fallback_cache: FallbackCache<TweetId, MediaFeature>,
}

#[derive(Default)]
pub(crate) struct TweetHydration {
    pub(crate) nullcast: TweetHydrationBatch<bool>,
    pub(crate) community: TweetHydrationBatch<i64>,
    pub(crate) nsfw_user: TweetHydrationBatch<bool>,
    pub(crate) nsfw_admin: TweetHydrationBatch<bool>,
    pub(crate) has_takedown: TweetHydrationBatch<bool>,
    pub(crate) takedown_reasons: TweetHydrationBatch<Vec<TakedownReason>>,
    pub(crate) edit_control: TweetHydrationBatch<EditControl>,
    pub(crate) media: TweetHydrationBatch<MediaFeature>,
}

impl TweetHydration {
    pub(crate) fn failed_entries(&self) -> usize {
        self.nullcast.failed_count()
            + self.community.failed_count()
            + self.nsfw_user.failed_count()
            + self.nsfw_admin.failed_count()
            + self.has_takedown.failed_count()
            + self.takedown_reasons.failed_count()
            + self.edit_control.failed_count()
            + self.media.failed_count()
    }
}

impl TesHydrator {
    pub(crate) fn new(
        tes_client: Arc<dyn TESClient + Send + Sync>,
        cache_mode: FallbackCacheMode,
    ) -> Self {
        Self {
            tes_client,
            // Preserve the existing media-cache policy; the bounded recovery behavior is
            // intentionally scoped to author restrictions in this change.
            fallback_cache: FallbackCache::new("media", CACHE_CAPACITY, cache_mode, None),
        }
    }

    pub async fn fetch_pure_core(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> HashMap<TweetId, PureCoreData> {
        if tweet_ids.is_empty() {
            return HashMap::new();
        }
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();
        record_batch_size(CLIENT, candidate_count_by_key.len());
        let fetched = timed_keyed_rpc(
            CLIENT,
            "get_tweet_core_datas",
            safety_level,
            &candidate_count_by_key,
            CLIENT_TIMEOUT,
            self.tes_client.get_tweet_core_datas(raw_ids),
        )
        .await;
        fetched
            .into_iter()
            .filter_map(|(id, r)| r.ok().flatten().map(|pcd| (TweetId(id), pcd)))
            .collect()
    }

    pub(crate) async fn hydrate_tweets(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> TweetHydration {
        let generation = self
            .fallback_cache
            .enabled()
            .then(|| self.fallback_cache.begin_request());
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();

        let (
            nullcast,
            community,
            nsfw_user,
            nsfw_admin,
            has_takedown,
            takedown_reasons,
            edit_control,
            media_entities,
        ) = tokio::join!(
            timed_results(
                CLIENT,
                "get_nullcast",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_nullcast(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_community",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_community(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_nsfw_user",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_nsfw_user(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_nsfw_admin",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_nsfw_admin(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_has_takedown",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_has_takedown(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_takedown_reasons",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_takedown_reasons(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_edit_control",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_edit_control(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_tweet_media_entities",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_tweet_media_entities(raw_ids.clone()),
            ),
        );

        let media = media_entities.map_keys(TweetId).map(media_feature);
        let media = if let Some(generation) = generation {
            self.fallback_cache
                .resolve_hydration_batch(generation, media)
        } else {
            media
        };

        TweetHydration {
            nullcast: nullcast.map_keys(TweetId),
            community: community.map_keys(TweetId),
            nsfw_user: nsfw_user.map_keys(TweetId),
            nsfw_admin: nsfw_admin.map_keys(TweetId),
            has_takedown: has_takedown.map_keys(TweetId),
            takedown_reasons: takedown_reasons.map_keys(TweetId),
            edit_control: edit_control.map_keys(TweetId),
            media,
        }
    }

    pub(crate) fn assemble_tweet_features(
        &self,
        candidates: &[TweetCandidateInput],
        core_datas: &HashMap<TweetId, PureCoreData>,
        tweet_keyed: &TweetHydration,
    ) -> HashMap<TweetId, TweetFeatures> {
        candidates
            .iter()
            .map(|c| {
                (
                    c.tweet_id,
                    build_tweet_features(c.tweet_id, core_datas, tweet_keyed),
                )
            })
            .collect()
    }
}

fn candidates_per_tweet(tweet_ids: &[TweetId]) -> HashMap<u64, usize> {
    let mut candidate_count_by_key = HashMap::with_capacity(tweet_ids.len());
    for tweet_id in tweet_ids {
        *candidate_count_by_key.entry(tweet_id.0).or_default() += 1;
    }
    candidate_count_by_key
}

fn build_tweet_features(
    tweet_id: TweetId,
    core_datas: &HashMap<TweetId, PureCoreData>,
    tweet_keyed: &TweetHydration,
) -> TweetFeatures {
    let id = tweet_id;

    let media = tweet_keyed.media.get_or_default(&id);
    let is_nullcast = tweet_keyed.nullcast.get(&id).copied().unwrap_or(false);
    let is_community_tweet = tweet_keyed.community.get(&id).is_some();
    let takedown = TakedownFeature {
        applied: tweet_keyed.has_takedown.get(&id).copied().unwrap_or(false),
        reasons: tweet_keyed.takedown_reasons.get_or_default(&id),
    };
    let nsfw = NsfwFeature {
        user: tweet_keyed.nsfw_user.get(&id).copied().unwrap_or(false),
        admin: tweet_keyed.nsfw_admin.get(&id).copied().unwrap_or(false),
    };
    let edit_control = tweet_keyed.edit_control.get(&id).cloned();

    core_datas
        .get(&tweet_id)
        .map(|core_data| TweetFeatures {
            core: CoreFeature {
                text: core_data.text.clone(),
                source_tweet_id: core_data.source_tweet_id,
                created_at_secs: core_data.created_at_secs,
            },
            media,
            takedown,
            nsfw,
            is_nullcast,
            is_community_tweet,
            edit_control,
        })
        .unwrap_or_default()
}

fn media_feature(entities: MediaEntities) -> MediaFeature {
    let mut feature = MediaFeature {
        has_media: !entities.is_empty(),
        ..Default::default()
    };

    for restrictions in entities
        .iter()
        .filter(|e| e.media_key.is_some())
        .filter_map(|e| e.additional_metadata.as_ref())
        .filter_map(|metadata| metadata.restrictions.as_ref())
    {
        feature.has_dmca_media |= restrictions.is_dmca == Some(true);
        if let Some(geo) = &restrictions.geo_restrictions {
            feature
                .geo_allow_list
                .extend(geo.whitelisted_country_codes.iter().flatten().cloned());
            feature
                .geo_deny_list
                .extend(geo.blacklisted_country_codes.iter().flatten().cloned());
        }
    }

    feature
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::Hydrated;
    use crate::models::{resolve_candidate, RawCandidate};
    use anyhow::Result;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use xai_core_entities::entities::{MediaEntity, PureCoreData};
    use xai_core_entities::tweet_entity_service_client::MockTESClient;
    use xai_x_thrift::media_information::{AdditionalMetadata, Restrictions};

    fn found<V>(id: u64, value: V) -> TweetHydrationBatch<V> {
        TweetHydrationBatch::from_results(
            [TweetId(id)],
            HashMap::from([(TweetId(id), Ok::<_, anyhow::Error>(Some(value)))]),
        )
    }

    fn candidate(tweet_id: u64, author_id: u64) -> TweetCandidateInput {
        let core = HashMap::from([(
            TweetId(tweet_id),
            PureCoreData {
                author_id,
                ..Default::default()
            },
        )]);
        resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(tweet_id),
                request_author_id: None,
            },
            &core,
        )
        .unwrap()
    }

    fn hydrator() -> TesHydrator {
        TesHydrator::new(
            Arc::new(MockTESClient::default()),
            FallbackCacheMode::Disabled,
        )
    }

    #[test]
    fn tweet_candidate_counts_deduplicate_backend_keys() {
        let tweet_ids = vec![TweetId(1), TweetId(1), TweetId(2)];

        assert_eq!(
            candidates_per_tweet(&tweet_ids),
            HashMap::from([(1, 2), (2, 1)])
        );
    }

    fn dmca_media_entity(has_media_key: bool) -> MediaEntity {
        MediaEntity {
            media_key: has_media_key.then(Default::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn assemble_flattens_media_geo_restrictions_from_tes() {
        use xai_core_entities::entities::MediaEntity;
        use xai_x_thrift::media_information::{AdditionalMetadata, GeoRestrictions, Restrictions};
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);
        let entity = |has_media_key: bool, allow: &[&str], deny: &[&str]| MediaEntity {
            media_key: has_media_key.then(Default::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(false),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(
                            allow.iter().map(|s| s.to_string()).collect(),
                        ),
                        blacklisted_country_codes: Some(
                            deny.iter().map(|s| s.to_string()).collect(),
                        ),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tweet_keyed = TweetHydration {
            media: found(
                10,
                media_feature(vec![
                    entity(true, &["us"], &["de"]),
                    entity(true, &["gb"], &["fr"]),
                    entity(false, &["ignored"], &["ignored"]),
                    MediaEntity::default(),
                ]),
            ),
            ..Default::default()
        };

        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &tweet_keyed);

        let f = &features[&TweetId(10)];
        assert_eq!(f.media.geo_allow_list, vec!["us", "gb"]);
        assert_eq!(f.media.geo_deny_list, vec!["de", "fr"]);
    }

    #[test]
    fn assemble_defaults_geo_lists_when_media_entities_missing() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);

        let features = hydrator().assemble_tweet_features(
            &candidates,
            &core_datas,
            &TweetHydration::default(),
        );

        let f = &features[&TweetId(10)];
        assert!(f.media.geo_allow_list.is_empty());
        assert!(f.media.geo_deny_list.is_empty());
    }

    #[test]
    fn assemble_hydrates_dmca_media() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);
        let tweet_keyed = TweetHydration {
            media: found(10, media_feature(vec![dmca_media_entity(true)])),
            ..Default::default()
        };

        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &tweet_keyed);

        assert!(features[&TweetId(10)].media.has_dmca_media);
        assert!(features[&TweetId(10)].media.has_media);
    }

    #[test]
    fn assemble_derives_has_media_from_media_entities() {
        let candidates = vec![candidate(10, 100), candidate(11, 100)];
        let core_datas = HashMap::from([
            (
                TweetId(10),
                PureCoreData {
                    author_id: 100,
                    ..Default::default()
                },
            ),
            (
                TweetId(11),
                PureCoreData {
                    author_id: 100,
                    ..Default::default()
                },
            ),
        ]);
        let tweet_keyed = TweetHydration {
            media: TweetHydrationBatch::from_results(
                [TweetId(10), TweetId(11)],
                HashMap::from([
                    (
                        TweetId(10),
                        Ok::<_, anyhow::Error>(Some(media_feature(vec![MediaEntity::default()]))),
                    ),
                    (
                        TweetId(11),
                        Ok::<_, anyhow::Error>(Some(media_feature(Vec::<MediaEntity>::new()))),
                    ),
                ]),
            ),
            ..Default::default()
        };

        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &tweet_keyed);

        assert!(features[&TweetId(10)].media.has_media);
        assert!(!features[&TweetId(11)].media.has_media);
    }

    #[test]
    fn dmca_metadata_without_media_key_is_ignored() {
        let feature = media_feature(vec![dmca_media_entity(false)]);
        assert!(!feature.has_dmca_media);
    }

    #[test]
    fn assemble_defaults_features_when_core_missing() {
        let candidates = vec![resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(10),
                request_author_id: Some(100),
            },
            &HashMap::new(),
        )
        .unwrap()];

        let features = hydrator().assemble_tweet_features(
            &candidates,
            &HashMap::new(),
            &TweetHydration::default(),
        );

        let f = &features[&TweetId(10)];
        assert!(f.core.text.is_empty());
        assert_eq!(f.core.created_at_secs, None);
        assert!(!f.media.has_media);
    }

    struct MediaFailingAfterFirstClient {
        inner: MockTESClient,
        media_calls: AtomicUsize,
    }

    impl MediaFailingAfterFirstClient {
        fn with_media(media_entities: HashMap<u64, Option<MediaEntities>>) -> Self {
            Self {
                inner: MockTESClient {
                    media_entities,
                    ..Default::default()
                },
                media_calls: AtomicUsize::new(0),
            }
        }
    }

    #[tonic::async_trait]
    impl TESClient for MediaFailingAfterFirstClient {
        async fn get_tweet_media_entities(
            &self,
            tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<MediaEntities>>> {
            if self.media_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.inner.get_tweet_media_entities(tweet_ids).await
            } else {
                tweet_ids
                    .into_iter()
                    .map(|id| (id, Err(anyhow::anyhow!("tes unavailable"))))
                    .collect()
            }
        }

        async fn get_nullcast(&self, tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<bool>>> {
            self.inner.get_nullcast(tweet_ids).await
        }

        async fn get_community(&self, tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<i64>>> {
            self.inner.get_community(tweet_ids).await
        }

        async fn get_nsfw_user(&self, tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<bool>>> {
            self.inner.get_nsfw_user(tweet_ids).await
        }

        async fn get_nsfw_admin(&self, tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<bool>>> {
            self.inner.get_nsfw_admin(tweet_ids).await
        }

        async fn get_has_takedown(
            &self,
            tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<bool>>> {
            self.inner.get_has_takedown(tweet_ids).await
        }

        async fn get_takedown_reasons(
            &self,
            tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<Vec<TakedownReason>>>> {
            self.inner.get_takedown_reasons(tweet_ids).await
        }

        async fn get_edit_control(
            &self,
            tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<EditControl>>> {
            self.inner.get_edit_control(tweet_ids).await
        }

        async fn get_tweet_core_datas(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<PureCoreData>>> {
            unreachable!()
        }

        async fn get_subscription_author_ids(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<u64>>> {
            unreachable!()
        }

        async fn get_quoted_tweets(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<xai_core_entities::entities::QuotedTweet>>> {
            unreachable!()
        }

        async fn get_reaction_context(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<xai_core_entities::entities::ReactionContext>>> {
            unreachable!()
        }

        async fn get_min_video_durations(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<i64>>> {
            unreachable!()
        }

        async fn get_media_count(&self, _tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<i64>>> {
            unreachable!()
        }

        async fn get_takedown_country_codes(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<Vec<String>>>> {
            unreachable!()
        }

        async fn get_language_code(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<String>>> {
            unreachable!()
        }

        async fn get_api_counts(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<xai_core_entities::entities::ApiCounts>>> {
            unreachable!()
        }

        async fn get_is_article(&self, _tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<bool>>> {
            unreachable!()
        }

        async fn get_is_premium(&self, _tweet_ids: Vec<u64>) -> HashMap<u64, Result<Option<bool>>> {
            unreachable!()
        }

        async fn get_urls(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<xai_core_entities::entities::UrlEntities>>> {
            unreachable!()
        }

        async fn get_exclusive_controls(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<xai_core_entities::entities::ExclusiveTweetControl>>>
        {
            unreachable!()
        }

        async fn get_trusted_friends_controls(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<xai_core_entities::entities::TrustedFriendsControl>>>
        {
            unreachable!()
        }

        async fn get_grok_post_ids(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<u64, Result<Option<String>>> {
            unreachable!()
        }

        async fn get_status_perspectives(
            &self,
            _tweet_ids: Vec<u64>,
            _metadata: Option<&tonic::metadata::MetadataMap>,
        ) -> HashMap<u64, Result<Option<xai_x_thrift::tweets::ApiPerspective>>> {
            unreachable!()
        }

        async fn get_api_media_entities(
            &self,
            _tweet_ids: Vec<u64>,
            _metadata: Option<&tonic::metadata::MetadataMap>,
        ) -> HashMap<u64, Result<Option<Vec<xai_x_thrift::entities::ApiMediaEntity>>>> {
            unreachable!()
        }

        async fn get_escherbird_entity_annotations(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> HashMap<
            u64,
            Result<Option<Vec<xai_core_entities::entities::EscherbirdEntityAnnotation>>>,
        > {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn media_stale_recovery_respects_cache_mode() {
        let tweet_ids = vec![TweetId(1)];
        for (mode, serves_stale) in [
            (FallbackCacheMode::ServeStale, true),
            (FallbackCacheMode::Shadow, false),
            (FallbackCacheMode::Disabled, false),
        ] {
            let hydrator = TesHydrator::new(
                Arc::new(MediaFailingAfterFirstClient::with_media(HashMap::from([(
                    1,
                    Some(vec![dmca_media_entity(true)]),
                )]))),
                mode,
            );
            let first = hydrator
                .hydrate_tweets(&tweet_ids, SafetyLevel::TimelineHome)
                .await;
            assert!(first.media.get_or_default(&TweetId(1)).has_dmca_media);

            let second = hydrator
                .hydrate_tweets(&tweet_ids, SafetyLevel::TimelineHome)
                .await;
            if serves_stale {
                let media = second.media.get_or_default(&TweetId(1));
                assert!(media.has_media);
                assert!(media.has_dmca_media);
            } else {
                assert!(matches!(
                    second.media.hydrated(&TweetId(1)),
                    Some(Hydrated::Failed(_))
                ));
                assert!(!second.media.get_or_default(&TweetId(1)).has_dmca_media);
            }
        }
    }

    #[tokio::test]
    async fn no_media_tweets_cache_default_entries_that_serve_stale() {
        let tweet_ids = vec![TweetId(1)];
        let hydrator = TesHydrator::new(
            Arc::new(MediaFailingAfterFirstClient::with_media(HashMap::from([(
                1,
                Some(Vec::new()),
            )]))),
            FallbackCacheMode::ServeStale,
        );

        let first = hydrator
            .hydrate_tweets(&tweet_ids, SafetyLevel::TimelineHome)
            .await;
        assert!(!first.media.get_or_default(&TweetId(1)).has_media);

        let second = hydrator
            .hydrate_tweets(&tweet_ids, SafetyLevel::TimelineHome)
            .await;
        assert!(matches!(
            second.media.hydrated(&TweetId(1)),
            Some(Hydrated::Found(_))
        ));
        assert!(!second.media.get_or_default(&TweetId(1)).has_media);
        assert_eq!(second.media.failed_count(), 0);
    }
}
