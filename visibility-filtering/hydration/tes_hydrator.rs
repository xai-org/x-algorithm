use crate::hydration::batch::TweetHydrationBatch;
use crate::hydration::metrics::{record_batch_size, timed_keyed_rpc, timed_results};
use crate::models::{
    CoreFeature, MediaFeature, NsfwFeature, TweetCandidateInput, TweetFeatures, TweetId,
};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use xai_core_entities::entities::{EditControl, MediaEntities, PureCoreData, TakedownReason};
use xai_core_entities::tweet_entity_service_client::TESClient;

const CLIENT_TIMEOUT: Duration = Duration::from_millis(150);
const CLIENT: &str = "tes";

pub struct TesHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

#[derive(Default)]
pub(crate) struct TweetHydration {
    pub(crate) nullcast: TweetHydrationBatch<bool>,
    pub(crate) community: TweetHydrationBatch<i64>,
    pub(crate) nsfw_user: TweetHydrationBatch<bool>,
    pub(crate) nsfw_admin: TweetHydrationBatch<bool>,
    pub(crate) takedown_reasons: TweetHydrationBatch<Vec<TakedownReason>>,
    pub(crate) edit_control: TweetHydrationBatch<EditControl>,
    pub(crate) media: TweetHydrationBatch<MediaFeature>,
}

impl TweetHydration {
    /// True when a Drop-sunk TES flag RPC failed. NotFound (tweet is not
    /// NSFW / not nullcast / no takedown) is not failure.
    pub(crate) fn safety_critical_failed(&self, id: &TweetId) -> bool {
        self.nullcast.is_failed(id)
            || self.nsfw_user.is_failed(id)
            || self.nsfw_admin.is_failed(id)
            || self.takedown_reasons.is_failed(id)
            || self.media.is_failed(id)
    }

    pub(crate) fn safety_hydration_failed(
        &self,
        id: &TweetId,
        core_datas: &HashMap<TweetId, PureCoreData>,
    ) -> bool {
        self.safety_critical_failed(id) || !core_datas.contains_key(id)
    }
}

impl TesHydrator {
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
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();

        let (
            nullcast,
            community,
            nsfw_user,
            nsfw_admin,
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

        TweetHydration {
            nullcast: nullcast.map_keys(TweetId),
            community: community.map_keys(TweetId),
            nsfw_user: nsfw_user.map_keys(TweetId),
            nsfw_admin: nsfw_admin.map_keys(TweetId),
            takedown_reasons: takedown_reasons.map_keys(TweetId),
            edit_control: edit_control.map_keys(TweetId),
            media: media_entities.map_keys(TweetId).map(media_feature),
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
    let takedown_reasons = tweet_keyed.takedown_reasons.get_or_default(&id);
    let nsfw = NsfwFeature {
        user: tweet_keyed.nsfw_user.get(&id).copied().unwrap_or(false),
        admin: tweet_keyed.nsfw_admin.get(&id).copied().unwrap_or(false),
    };
    let edit_control = tweet_keyed.edit_control.get(&id).cloned();
    let core = core_datas
        .get(&tweet_id)
        .map(|core_data| CoreFeature {
            text: core_data.text.clone(),
            source_tweet_id: core_data.source_tweet_id,
        })
        .unwrap_or_default();

    TweetFeatures {
        core,
        media,
        takedown_reasons,
        nsfw,
        is_nullcast,
        is_community_tweet,
        edit_control,
    }
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
    use crate::models::{resolve_candidate, RawCandidate};
    use xai_core_entities::entities::{MediaEntity, PureCoreData};
    use xai_core_entities::tweet_entity_service_client::MockTESClient;
    use xai_x_thrift::media_information::{AdditionalMetadata, Restrictions};

    fn found<V>(id: u64, value: V) -> TweetHydrationBatch<V> {
        TweetHydrationBatch::from_results(
            [TweetId(id)],
            HashMap::from([(TweetId(id), Ok::<_, anyhow::Error>(Some(value)))]),
        )
    }

    fn rpc_failed<V>(id: u64) -> TweetHydrationBatch<V> {
        TweetHydrationBatch::from_results(
            [TweetId(id)],
            HashMap::from([(
                TweetId(id),
                Err::<Option<V>, _>("tes timeout"),
            )]),
        )
    }

    fn not_found<V>(id: u64) -> TweetHydrationBatch<V> {
        TweetHydrationBatch::from_results(
            [TweetId(id)],
            HashMap::from([(TweetId(id), Ok::<Option<V>, anyhow::Error>(None))]),
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
        TesHydrator {
            tes_client: Arc::new(MockTESClient::default()),
        }
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
    fn assemble_hydrates_text_from_core_data() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                text: "muted words".to_string(),
                ..Default::default()
            },
        )]);

        let features = hydrator().assemble_tweet_features(
            &candidates,
            &core_datas,
            &TweetHydration::default(),
        );

        assert_eq!(features[&TweetId(10)].core.text, "muted words");
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
        assert!(!f.media.has_media);
    }

    #[test]
    fn tes_nsfw_flag_rpc_failure_is_safety_critical() {
        let keyed = TweetHydration {
            nsfw_user: rpc_failed(10),
            ..Default::default()
        };
        assert!(keyed.safety_critical_failed(&TweetId(10)));
        assert!(!keyed.safety_critical_failed(&TweetId(11)));
    }

    #[test]
    fn tes_nsfw_flag_not_found_is_not_safety_critical() {
        let keyed = TweetHydration {
            nsfw_user: not_found(10),
            nsfw_admin: not_found(10),
            nullcast: not_found(10),
            ..Default::default()
        };
        assert!(!keyed.safety_critical_failed(&TweetId(10)));
    }

    #[test]
    fn tes_nsfw_flag_rpc_failure_still_assembles_false_bits() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(TweetId(10), PureCoreData {
            author_id: 100,
            ..Default::default()
        })]);
        let keyed = TweetHydration {
            nsfw_user: rpc_failed(10),
            ..Default::default()
        };
        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &keyed);
        assert!(!features[&TweetId(10)].nsfw.user);
        assert!(keyed.safety_critical_failed(&TweetId(10)));
    }

    #[test]
    fn assemble_keeps_found_nsfw_flag_when_core_missing() {
        let candidates = vec![resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(10),
                request_author_id: Some(100),
            },
            &HashMap::new(),
        )
        .unwrap()];
        let keyed = TweetHydration {
            nsfw_user: found(10, true),
            ..Default::default()
        };
        let features = hydrator().assemble_tweet_features(&candidates, &HashMap::new(), &keyed);
        assert!(features[&TweetId(10)].nsfw.user);
        assert!(features[&TweetId(10)].core.text.is_empty());
        assert!(keyed.safety_hydration_failed(&TweetId(10), &HashMap::new()));
    }

    #[test]
    fn core_present_and_flags_not_found_is_not_safety_hydration_failed() {
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);
        let keyed = TweetHydration {
            nsfw_user: not_found(10),
            ..Default::default()
        };
        assert!(!keyed.safety_hydration_failed(&TweetId(10), &core_datas));
    }
}
