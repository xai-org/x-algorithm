use crate::clients::strato;
use crate::models::{Deletion, MediaFeature, NsfwFeature, ReactionContext, TweetFeatures};
use crate::thrift_read::{read_fields, skip_elements, SKIP_DEPTH};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use thrift::protocol::{TInputProtocol, TType};
use xai_core_entities::entities::{
    EditControl, ExclusiveTweetControl, TakedownReason, TrustedFriendsControl,
};
use xai_strato::strato_thrift::{strato_decode, MValProtocol, StratoRead};
use xai_strato::{encode, Bytes, MValCodec, StratoGrpc};

const COLUMN: &str = "tweetypie/federated/tweetForVisibility.Tweet";
const STATE_COLUMN: &str = "tweetypie/federated/tweetState.Tweet";

pub(crate) struct TweetSource {
    pub(crate) grpc_client: Arc<StratoGrpc>,
}

impl TweetSource {
    pub(crate) async fn get_tweet_values(&self, tweet_ids: &[u64]) -> HashMap<u64, Result<Bytes>> {
        self.fetch(COLUMN, tweet_ids).await
    }

    pub(crate) async fn get_tweet_states(&self, tweet_ids: &[u64]) -> HashMap<u64, Result<Bytes>> {
        self.fetch(STATE_COLUMN, tweet_ids).await
    }

    async fn fetch(&self, column: &str, tweet_ids: &[u64]) -> HashMap<u64, Result<Bytes>> {
        let keys = tweet_ids
            .iter()
            .map(|tweet_id| (column, encode(&(*tweet_id, ()))));
        let result_batch = strato::fetch_each(&self.grpc_client, keys, None).await;
        tweet_ids.iter().copied().zip(result_batch).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum TweetState {
    Found,
    NotFound,
    Deleted,
    BounceDeleted,
}

impl TweetState {
    pub(crate) fn deletion(self) -> Option<Deletion> {
        match self {
            Self::Deleted => Some(Deletion::Deleted),
            Self::BounceDeleted => Some(Deletion::BounceDeleted),
            Self::Found | Self::NotFound => None,
        }
    }
}

struct StatusState(i32);

impl StratoRead for StatusState {
    fn read(proto: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        proto.read_i32().map(Self)
    }
}

pub(crate) fn decode_tweet_state(bytes: &[u8]) -> Result<Option<TweetState>> {
    let Some(StatusState(state)) = strato::row(strato_decode(bytes)?)? else {
        return Ok(None);
    };
    Ok(Some(match state {
        0 => TweetState::Found,
        1 => TweetState::NotFound,
        14 => TweetState::Deleted,
        15 => TweetState::BounceDeleted,
        other => bail!("unexpected StatusState {other}"),
    }))
}

pub(crate) fn decode_tweet(bytes: &[u8]) -> Result<Option<TweetFeatures>> {
    let result = std::panic::catch_unwind(|| strato_decode::<Tweet>(bytes))
        .map_err(|_| anyhow!("MVal decoder panicked"))?
        .map_err(|e| anyhow!("MVal decode error: {e}"))?;
    let Some(tweet) = strato::row(result)? else {
        return Ok(None);
    };
    tweet
        .project()
        .map(Some)
        .ok_or_else(|| anyhow!("tweet value without coreData"))
}

#[derive(Default)]
struct Tweet {
    core_data: Option<CoreData>,
    media: Vec<Media>,
    takedown_reasons: Vec<TakedownReason>,
    community_id: Option<NonZeroU64>,
    exclusive_tweet_control: Option<ExclusiveTweetControl>,
    trusted_friends_list_id: Option<u64>,
    edit_control: Option<EditControl>,
    has_media_refs: bool,
    has_media_keys: bool,
    has_card_reference: bool,
    article_id: Option<NonZeroU64>,
    narrowcast_place_id: Option<u64>,
    reaction_context: Option<ReactionContext>,
}

#[derive(Default)]
struct CoreData {
    nsfw_user: bool,
    nsfw_admin: bool,
    nullcast: bool,
}

#[derive(Default)]
struct Media {
    has_media_key: bool,
    restrictions: Option<MediaRestrictions>,
}

#[derive(Default)]
struct MediaRestrictions {
    is_dmca: bool,
    geo_allow_list: Vec<String>,
    geo_deny_list: Vec<String>,
}

impl Tweet {
    fn project(self) -> Option<TweetFeatures> {
        let core_data = self.core_data?;
        Some(TweetFeatures {
            is_nullcast: core_data.nullcast,
            nsfw: NsfwFeature {
                user: core_data.nsfw_user,
                admin: core_data.nsfw_admin,
            },
            takedown_reasons: self.takedown_reasons,
            media: MediaFeature {
                has_media: self.has_media_refs || self.has_card_reference,
                has_uploaded_media: self.has_media_keys,
                ..media_feature(self.media)
            },
            community_id: self.community_id,
            trusted_friends_list_id: self.trusted_friends_list_id,
            latest_edit_tweet_id: self.edit_control.as_ref().and_then(latest_edit_tweet_id),
            exclusive_conversation_author_id: self
                .exclusive_tweet_control
                .map(|control| control.conversation_author_id),
            article_id: self.article_id,
            narrowcast_place_id: self.narrowcast_place_id,
            reaction_context: self.reaction_context,
        })
    }
}

fn latest_edit_tweet_id(control: &EditControl) -> Option<u64> {
    let initial = match control {
        EditControl::Initial(initial) => initial,
        EditControl::Edit(edit) => edit.edit_control_initial.as_ref()?,
    };
    initial.edit_tweet_ids.last().copied()
}

fn media_feature(entities: Vec<Media>) -> MediaFeature {
    let mut feature = MediaFeature::default();
    for restrictions in entities
        .into_iter()
        .filter(|e| e.has_media_key)
        .filter_map(|e| e.restrictions)
    {
        feature.has_dmca_media |= restrictions.is_dmca;
        feature.geo_allow_list.extend(restrictions.geo_allow_list);
        feature.geo_deny_list.extend(restrictions.geo_deny_list);
    }
    feature
}

impl StratoRead for Tweet {
    fn read(proto: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        let mut tweet = Tweet::default();
        read_fields(proto, |proto, field| {
            match field.id {
                Some(2) => tweet.core_data = Some(read_core_data(proto)?),
                Some(7) => {
                    let list = proto.read_list_begin()?;
                    let mut media = Vec::with_capacity(usize::try_from(list.size).unwrap_or(0));
                    for _ in 0..list.size {
                        media.push(read_media(proto)?);
                    }
                    proto.read_list_end()?;
                    tweet.media = media;
                }
                Some(30) => tweet.takedown_reasons = Vec::<TakedownReason>::from_thrift(proto),
                Some(118) => {
                    tweet.has_card_reference = true;
                    return Ok(false);
                }
                Some(125) => tweet.community_id = read_first_community_id(proto)?,
                Some(155) => {
                    tweet.exclusive_tweet_control = Some(ExclusiveTweetControl::from_thrift(proto));
                }
                Some(156) => {
                    tweet.trusted_friends_list_id =
                        Some(TrustedFriendsControl::from_thrift(proto).trusted_friends_list_id);
                }
                Some(157) => tweet.edit_control = Some(EditControl::from_thrift(proto)),
                Some(162) => tweet.has_media_refs = skip_list_non_empty(proto)?,
                Some(170) => {
                    tweet.article_id = read_id_field(proto, |proto| {
                        Ok(NonZeroU64::new(proto.read_i64()?.cast_unsigned()))
                    })?;
                }
                Some(172) => {
                    tweet.narrowcast_place_id = read_id_field(proto, |proto| {
                        Ok(u64::from_str_radix(&proto.read_string()?, 16).ok())
                    })?;
                }
                Some(178) if field.field_type == TType::Struct => {
                    tweet.reaction_context = read_reaction_context(proto)?;
                }
                Some(32766) => tweet.has_media_keys = skip_list_non_empty(proto)?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        Ok(tweet)
    }
}

fn read_core_data(proto: &mut MValProtocol<'_>) -> thrift::Result<CoreData> {
    let mut core_data = CoreData::default();
    read_fields(proto, |proto, field| {
        match field.id {
            Some(9) => core_data.nsfw_user = proto.read_bool()?,
            Some(10) => core_data.nsfw_admin = proto.read_bool()?,
            Some(11) => core_data.nullcast = proto.read_bool()?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(core_data)
}

fn read_media(proto: &mut MValProtocol<'_>) -> thrift::Result<Media> {
    let mut media = Media::default();
    read_fields(proto, |proto, field| {
        match field.id {
            Some(21) => {
                media.has_media_key = true;
                return Ok(false);
            }
            Some(22) => media.restrictions = read_additional_metadata_restrictions(proto)?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(media)
}

fn read_additional_metadata_restrictions(
    proto: &mut MValProtocol<'_>,
) -> thrift::Result<Option<MediaRestrictions>> {
    let mut restrictions = None;
    read_fields(proto, |proto, field| {
        if field.id != Some(12) {
            return Ok(false);
        }
        restrictions = Some(read_restrictions(proto)?);
        Ok(true)
    })?;
    Ok(restrictions)
}

fn read_restrictions(proto: &mut MValProtocol<'_>) -> thrift::Result<MediaRestrictions> {
    let mut restrictions = MediaRestrictions::default();
    read_fields(proto, |proto, field| {
        match field.id {
            Some(1) => restrictions.is_dmca = proto.read_bool()?,
            Some(3) => {
                (restrictions.geo_allow_list, restrictions.geo_deny_list) =
                    read_geo_restrictions(proto)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(restrictions)
}

fn read_geo_restrictions(
    proto: &mut MValProtocol<'_>,
) -> thrift::Result<(Vec<String>, Vec<String>)> {
    let (mut allow, mut deny) = (Vec::new(), Vec::new());
    read_fields(proto, |proto, field| {
        match field.id {
            Some(1) => allow = read_string_list(proto)?,
            Some(2) => deny = read_string_list(proto)?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok((allow, deny))
}

fn read_string_list(proto: &mut MValProtocol<'_>) -> thrift::Result<Vec<String>> {
    let list = proto.read_list_begin()?;
    let mut strings = Vec::with_capacity(usize::try_from(list.size).unwrap_or(0));
    for _ in 0..list.size {
        strings.push(proto.read_string()?);
    }
    proto.read_list_end()?;
    Ok(strings)
}

fn skip_list_non_empty(proto: &mut MValProtocol<'_>) -> thrift::Result<bool> {
    let list = proto.read_list_begin()?;
    skip_elements(proto, &[list.element_type], list.size, SKIP_DEPTH)?;
    proto.read_list_end()?;
    Ok(list.size > 0)
}

fn read_first_community_id(proto: &mut MValProtocol<'_>) -> thrift::Result<Option<NonZeroU64>> {
    let mut first = None;
    read_fields(proto, |proto, field| {
        if field.id != Some(1) {
            return Ok(false);
        }
        let list = proto.read_list_begin()?;
        if list.size > 0 {
            first = NonZeroU64::new(proto.read_i64()?.cast_unsigned());
            skip_elements(proto, &[list.element_type], list.size - 1, SKIP_DEPTH)?;
        }
        proto.read_list_end()?;
        Ok(true)
    })?;
    Ok(first)
}

fn read_reaction_context(proto: &mut MValProtocol<'_>) -> thrift::Result<Option<ReactionContext>> {
    let (mut post, mut user) = (None, None);
    read_fields(proto, |proto, field| {
        match (field.id, field.field_type) {
            (Some(1), TType::I64) => post = Some(proto.read_i64()?.cast_unsigned()),
            (Some(2), TType::I64) => user = Some(proto.read_i64()?.cast_unsigned()),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(post
        .zip(user)
        .map(|(reacted_to_post_id, reacted_to_user_id)| ReactionContext {
            reacted_to_post_id,
            reacted_to_user_id,
        }))
}

fn read_id_field<T>(
    proto: &mut MValProtocol<'_>,
    read: impl Fn(&mut MValProtocol<'_>) -> thrift::Result<Option<T>>,
) -> thrift::Result<Option<T>> {
    let mut id = None;
    read_fields(proto, |proto, field| {
        if field.id != Some(1) {
            return Ok(false);
        }
        id = read(proto)?;
        Ok(true)
    })?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::Thrift::{self, Bool, List, Str, Struct, I64};
    use crate::clients::strato::answer::{found, no_row};
    use crate::thrift_read::tests::assert_skips_like_thrift;
    use std::collections::BTreeSet;
    use thrift::protocol::TType;
    use thrift::OrderedFloat;
    use xai_core_entities::entities::{EditControlEdit, EditControlInitial, MediaEntity};
    use xai_x_thrift::media_common::{MediaCategory, MediaKey, OwnershipInfo};
    use xai_x_thrift::media_entity::ColorValue;
    use xai_x_thrift::media_information::{
        AdditionalMetadata, AltText, CalibrationBucket, CalibrationLevel, CandidateCropsResponse,
        ClassInfo, ClassificationInfo, ComputationStatus, ContentIdInfo, DomainRestrictions, Face,
        FaceResponse, GeoRestrictions, MediaAvailability, MediaAvailabilityStatus, Rect,
        Restrictions,
    };
    use xai_x_thrift::tweet_media::{MediaResizeMethod, MediaSize, MediaSizeType};

    const TWEET_ID: i64 = 10;
    const AUTHOR_ID: i64 = 7001;
    const CONVERSATION_AUTHOR_ID: i64 = 7003;
    const TRUSTED_FRIENDS_LIST_ID: i64 = 9001;
    const ARTICLE_ID: i64 = 9002;
    const PHOTO_REPLY_ID: i64 = 20_000;
    const REACTED_TO_POST_ID: i64 = 9003;
    const REACTED_TO_USER_ID: i64 = 7004;

    fn exclusive_conversation() -> (i16, Thrift) {
        (155, Struct(vec![(1, I64(CONVERSATION_AUTHOR_ID))]))
    }

    fn narrowcast_place(id: &str) -> (i16, Thrift) {
        (172, Struct(vec![(1, Str(id.into()))]))
    }

    #[test]
    fn a_tweet_state_decodes_from_its_status_state_value() {
        use TweetState::{BounceDeleted, Deleted, Found, NotFound};
        for (value, state) in [
            (0, Found),
            (1, NotFound),
            (14, Deleted),
            (15, BounceDeleted),
        ] {
            let decoded = decode_tweet_state(&found(Thrift::I32(value))).unwrap();
            assert_eq!(decoded, Some(state), "{value}");
        }
        assert_eq!(decode_tweet_state(&no_row()).unwrap(), None);
        let failed = 4;
        assert!(decode_tweet_state(&found(Thrift::I32(failed))).is_err());
    }

    fn encode_tweet(core_data: Vec<(i16, Thrift)>, rest: Vec<(i16, Thrift)>) -> Vec<u8> {
        let core_data = [(1, I64(AUTHOR_ID)), (2, Str("tweet text".into()))]
            .into_iter()
            .chain(core_data);
        let tweet = [(1, I64(TWEET_ID)), (2, Struct(core_data.collect()))]
            .into_iter()
            .chain(rest);
        found(Struct(tweet.collect()))
    }

    fn non_empty_list(non_empty: bool) -> Thrift {
        List(
            TType::Struct,
            non_empty.then(|| Struct(vec![])).into_iter().collect(),
        )
    }

    fn media_fixture(
        has_refs: bool,
        has_card: bool,
        keys: Option<bool>,
        entities: &[MediaEntity],
    ) -> Vec<u8> {
        let entities = entities.iter().cloned().map(Thrift::generated);
        let fields = [(7, List(TType::Struct, entities.collect()))]
            .into_iter()
            .chain(has_card.then(|| (118, Struct(vec![]))))
            .chain([(162, non_empty_list(has_refs))])
            .chain(keys.map(|non_empty| (32766, non_empty_list(non_empty))));
        encode_tweet(vec![], fields.collect())
    }

    fn restricted_media(has_key: bool, dmca: bool, allow: &[&str], deny: &[&str]) -> MediaEntity {
        MediaEntity {
            media_key: has_key.then(MediaKey::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(dmca),
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
        }
    }

    fn decode_fixture(bytes: &[u8]) -> TweetFeatures {
        decode_tweet(bytes).unwrap().unwrap()
    }

    fn media_id(tweet_id: i64, index: i64) -> i64 {
        tweet_id - 1_000 + index
    }

    fn photo(tweet_id: i64, index: i64, geo_allow_list: Option<&[&str]>) -> MediaEntity {
        let media_id = media_id(tweet_id, index);
        let size = |size_type, width, height| MediaSize {
            size_type: Some(MediaSizeType(size_type)),
            resize_method: Some(MediaResizeMethod(i32::from(size_type == 0))),
            width: Some(width),
            height: Some(height),
            ..Default::default()
        };
        let class = |label: &str| ClassInfo {
            label: Some(label.to_string()),
            probability: Some(OrderedFloat(0.0125)),
            calibration_buckets: Some(
                (1..=3)
                    .map(|level| CalibrationBucket {
                        level: Some(CalibrationLevel(level)),
                        threshold: Some(OrderedFloat(0.25 * f64::from(level))),
                        precision: Some(OrderedFloat(0.9)),
                        recall: Some(OrderedFloat(0.5)),
                    })
                    .collect(),
            ),
            model_version: Some("model_8C_000001".to_string()),
            ..Default::default()
        };
        let face = |left| Face {
            bounding_box: Some(Rect {
                left: Some(left),
                top: Some(60),
                width: Some(142),
                height: Some(142),
            }),
        };
        MediaEntity {
            from_index: Some(186),
            to_index: Some(209),
            url: Some("https://t.co/Synthetic1".to_string()),
            display_url: Some("pic.x.com/Synthetic1".to_string()),
            media_url: Some(format!("http://media.example.com/{media_id}.jpg")),
            media_url_https: Some(format!("https://media.example.com/{media_id}.jpg")),
            expanded_url: Some(format!(
                "https://x.example.com/synthetic_user/status/{tweet_id}/photo/{index}"
            )),
            media_id: Some(media_id),
            nsfw: Some(false),
            sizes: Some(BTreeSet::from([
                size(0, 150, 150),
                size(1, 680, 510),
                size(2, 1200, 900),
                size(3, 2048, 1536),
            ])),
            media_path: Some(format!("media/{media_id}.jpg")),
            media_key: Some(MediaKey {
                media_category: Some(MediaCategory::TWEET_IMAGE),
                media_id: Some(media_id),
            }),
            additional_metadata: Some(AdditionalMetadata {
                face_data: Some(FaceResponse {
                    status: Some(ComputationStatus::COMPLETE),
                    faces: Some(
                        ["large", "medium", "small", "orig"]
                            .into_iter()
                            .zip([32, 24, 12, 48])
                            .map(|(size, left)| (size.to_string(), vec![face(left)]))
                            .collect(),
                    ),
                }),
                classifications: Some(ClassificationInfo {
                    probabilities: Some(
                        [
                            "IN", "NU", "SX", "GR", "VI", "DR", "WE", "HA", "SP", "MI", "OK",
                        ]
                        .into_iter()
                        .map(class)
                        .collect(),
                    ),
                    predictions: None,
                }),
                ownership_info: Some(OwnershipInfo {
                    user_id: Some(AUTHOR_ID),
                    tweet_id: Some(tweet_id),
                    ..Default::default()
                }),
                candidate_crops: Some(CandidateCropsResponse {
                    status: Some(ComputationStatus::COMPLETE),
                    ..Default::default()
                }),
                media_availability: Some(MediaAvailability {
                    status: MediaAvailabilityStatus::AVAILABLE,
                    reason: None,
                    unavailability_info: None,
                }),
                restrictions: geo_allow_list.map(|allow| Restrictions {
                    is_dmca: Some(false),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(
                            allow.iter().map(|s| s.to_string()).collect(),
                        ),
                        blacklisted_country_codes: Some(vec![]),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn photo_reply(index: i64) -> (Vec<u8>, TweetFeatures) {
        let tweet_id = PHOTO_REPLY_ID + index;
        let editable_until_msecs = 1_729_000_000 * 1_000 + 3_600_000;
        let photos = 2 + index % 3;
        let geo_allow_list: Option<&[&str]> = (index % 5 == 0).then_some(&["us", "ca"]);
        let text = "Synthetic photo reply text for the decode fixture, as long as a typical post \
                    body \u{1f415}\u{1f30a} @synthetic_parent @synthetic_friend \
                    #synthetic #fixture https://t.co/Synthetic1 https://t.co/Synthetic1";
        let core_data = vec![
            (1, I64(AUTHOR_ID)),
            (2, Str(text.into())),
            (3, Str("synthetic client".into())),
            (4, I64(1_729_000_000)),
            (
                5,
                Struct(vec![
                    (1, I64(tweet_id - 1)),
                    (2, I64(AUTHOR_ID + 1)),
                    (3, Str("synthetic_parent".into())),
                ]),
            ),
            (8, Bool(false)),
            (9, Bool(false)),
            (10, Bool(false)),
            (11, Bool(false)),
            (13, I64(1)),
            (14, I64(tweet_id - 1)),
            (15, Bool(true)),
            (19, Str("0123456789abcdef".into())),
        ];
        let annotation = |group| {
            Struct(vec![
                (1, I64(group)),
                (2, I64(group * 1_000)),
                (3, I64(group * 1_000_000)),
            ])
        };
        let media_ref = |i| {
            Struct(vec![
                (1, Str(format!("3_{}", media_id(tweet_id, i)))),
                (2, I64(tweet_id)),
                (3, I64(AUTHOR_ID)),
            ])
        };
        let media_key = |i| {
            Thrift::generated(MediaKey {
                media_category: Some(MediaCategory::TWEET_IMAGE),
                media_id: Some(media_id(tweet_id, i)),
            })
        };
        let edit = Struct(vec![
            (1, List(TType::I64, vec![I64(tweet_id)])),
            (2, I64(editable_until_msecs)),
            (3, I64(5)),
            (4, Bool(true)),
        ]);
        let bytes = found(Struct(vec![
            (1, I64(tweet_id)),
            (2, Struct(core_data)),
            (
                7,
                List(
                    TType::Struct,
                    (0..photos)
                        .map(|i| {
                            Thrift::generated(photo(tweet_id, i, geo_allow_list.filter(|_| i == 0)))
                        })
                        .collect(),
                ),
            ),
            (30, List(TType::Struct, vec![])),
            (
                106,
                Struct(vec![(
                    1,
                    List(TType::Struct, (1..=5).map(annotation).collect()),
                )]),
            ),
            (157, Struct(vec![(1, edit)])),
            (
                162,
                List(TType::Struct, (0..photos).map(media_ref).collect()),
            ),
            (
                32766,
                List(TType::Struct, (0..photos).map(media_key).collect()),
            ),
        ]));
        let features = TweetFeatures {
            media: MediaFeature {
                has_media: true,
                has_uploaded_media: true,
                geo_allow_list: geo_allow_list
                    .unwrap_or_default()
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                ..Default::default()
            },
            latest_edit_tweet_id: Some(tweet_id.cast_unsigned()),
            ..Default::default()
        };
        (bytes, features)
    }

    #[test]
    fn photo_replies_project_their_features() {
        for (bytes, features) in (0..10).map(photo_reply) {
            assert_eq!(decode_fixture(&bytes), features);
        }
    }

    #[test]
    fn skip_ends_where_thrift_does_and_fails_as_it_does_on_truncated_or_corrupt_bytes() {
        let (bytes, _) = photo_reply(0);
        for end in 0..=bytes.len() {
            assert_skips_like_thrift(&bytes[..end], TType::Struct);
        }
        for at in 0..bytes.len() {
            for byte in [0x00, 0x01, 0x05, 0x0f, 0x10, 0x7f, 0x80, 0xff] {
                let mut corrupt = bytes.clone();
                corrupt[at] = byte;
                assert_skips_like_thrift(&corrupt, TType::Struct);
            }
        }
    }

    #[test]
    fn projects_every_read_field_of_a_found_tweet() {
        let media = Thrift::generated(restricted_media(true, true, &[], &[]));
        let edit = Struct(vec![
            (1, List(TType::I64, vec![I64(TWEET_ID)])),
            (3, I64(4)),
        ]);
        let bytes = encode_tweet(
            vec![(9, Bool(true)), (10, Bool(false)), (11, Bool(true))],
            vec![
                (7, List(TType::Struct, vec![media])),
                (
                    30,
                    List(TType::Struct, vec![Struct(vec![(4, Struct(vec![]))])]),
                ),
                (118, Struct(vec![])),
                (125, Struct(vec![(1, List(TType::I64, vec![I64(500)]))])),
                exclusive_conversation(),
                (156, Struct(vec![(1, I64(TRUSTED_FRIENDS_LIST_ID))])),
                (157, Struct(vec![(1, edit)])),
                (162, non_empty_list(true)),
                (170, Struct(vec![(1, I64(ARTICLE_ID))])),
                narrowcast_place("7f00000000000001"),
                (
                    178,
                    Struct(vec![
                        (1, I64(REACTED_TO_POST_ID)),
                        (2, I64(REACTED_TO_USER_ID)),
                    ]),
                ),
                (32766, non_empty_list(true)),
            ],
        );
        let media = MediaFeature {
            has_media: true,
            has_uploaded_media: true,
            has_dmca_media: true,
            ..Default::default()
        };

        assert_eq!(
            decode_fixture(&bytes),
            TweetFeatures {
                media,
                takedown_reasons: vec![TakedownReason::Dmca],
                nsfw: NsfwFeature {
                    user: true,
                    admin: false
                },
                is_nullcast: true,
                community_id: NonZeroU64::new(500),
                trusted_friends_list_id: Some(TRUSTED_FRIENDS_LIST_ID as u64),
                latest_edit_tweet_id: Some(TWEET_ID as u64),
                exclusive_conversation_author_id: Some(CONVERSATION_AUTHOR_ID as u64),
                article_id: NonZeroU64::new(ARTICLE_ID as u64),
                narrowcast_place_id: Some(0x7f00_0000_0000_0001),
                reaction_context: Some(ReactionContext {
                    reacted_to_post_id: REACTED_TO_POST_ID as u64,
                    reacted_to_user_id: REACTED_TO_USER_ID as u64,
                }),
            }
        );
    }

    #[test]
    fn a_reaction_context_missing_either_id_or_of_another_wire_type_is_no_reaction() {
        let post = || (1, I64(REACTED_TO_POST_ID));
        let user = || (2, I64(REACTED_TO_USER_ID));
        for (name, context) in [
            ("post only", Struct(vec![post()])),
            ("user only", Struct(vec![user()])),
            ("neither", Struct(vec![])),
            ("an i64 context", I64(REACTED_TO_POST_ID)),
            ("a string context", Str("reaction".into())),
            (
                "a list context",
                List(TType::Struct, vec![Struct(vec![post(), user()])]),
            ),
            (
                "a string post id",
                Struct(vec![(1, Str("7".into())), user()]),
            ),
            (
                "a struct user id",
                Struct(vec![post(), (2, Struct(vec![user()]))]),
            ),
        ] {
            let bytes = encode_tweet(vec![], vec![(178, context), (32766, non_empty_list(true))]);

            let features = decode_fixture(&bytes);
            assert_eq!(features.reaction_context, None, "{name}");
            assert!(
                features.media.has_uploaded_media,
                "{name}: the field after the reaction context did not decode"
            );
        }
    }

    #[test]
    fn a_narrowcast_place_keeps_all_64_bits_of_its_hex_and_a_non_hex_one_is_no_place() {
        for (id, expected) in [
            ("a000000000000001", Some(0xa000_0000_0000_0001)),
            ("not hex", None),
        ] {
            let bytes = encode_tweet(
                vec![],
                vec![narrowcast_place(id), (32766, non_empty_list(true))],
            );

            let features = decode_fixture(&bytes);
            assert_eq!(features.narrowcast_place_id, expected, "{id}");
            assert!(
                features.media.has_uploaded_media,
                "{id}: the field after the place did not decode"
            );
        }
    }

    #[test]
    fn an_article_with_id_0_is_no_article() {
        let bytes = encode_tweet(vec![], vec![(170, Struct(vec![(1, I64(0))]))]);

        assert_eq!(decode_fixture(&bytes).article_id, None);
    }

    #[test]
    fn plain_tweet_decodes_default_features() {
        let bytes = encode_tweet(vec![], vec![]);

        assert_eq!(decode_fixture(&bytes), TweetFeatures::default());
    }

    #[test]
    fn media_restrictions_fold_only_across_keyed_entities() {
        let bytes = media_fixture(
            true,
            false,
            Some(true),
            &[
                restricted_media(true, true, &["us"], &["de"]),
                restricted_media(true, false, &["gb"], &["fr"]),
                restricted_media(false, true, &["ignored"], &["ignored"]),
                MediaEntity::default(),
            ],
        );

        assert_eq!(
            decode_fixture(&bytes).media,
            MediaFeature {
                has_media: true,
                has_uploaded_media: true,
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string(), "gb".to_string()],
                geo_deny_list: vec!["de".to_string(), "fr".to_string()],
            }
        );
    }

    #[test]
    fn media_fields_vf_does_not_read_are_skipped() {
        let classified = MediaEntity {
            url: Some("https://t.co/x".to_string()),
            dominant_color_grid: Some(vec![ColorValue::default(); 3]),
            media_key: Some(MediaKey::default()),
            additional_metadata: Some(AdditionalMetadata {
                alt_text: Some(AltText {
                    text: Some("a cat".to_string()),
                }),
                classifications: Some(ClassificationInfo {
                    probabilities: Some(vec![ClassInfo {
                        label: Some("nsfw".to_string()),
                        calibration_buckets: Some(vec![CalibrationBucket::default(); 2]),
                        model_version: Some("v7".to_string()),
                        ..Default::default()
                    }]),
                    predictions: Some(vec![ClassInfo::default()]),
                }),
                restrictions: Some(Restrictions {
                    is_dmca: Some(true),
                    domain_restrictions: Some(DomainRestrictions {
                        whitelist: Some(vec!["x.com".to_string()]),
                    }),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(vec!["us".to_string()]),
                        blacklisted_country_codes: Some(vec!["de".to_string()]),
                    }),
                    content_id_info: Some(ContentIdInfo {
                        copyright_holder_name: Some("holder".to_string()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let unknown = || {
            let element = Struct(vec![(2, Str("x".into()))]);
            (900, Struct(vec![(1, List(TType::Struct, vec![element]))]))
        };
        let deny = (2, List(TType::String, vec![Str("fr".into())]));
        let restrictions = Struct(vec![unknown(), (3, Struct(vec![unknown(), deny]))]);
        let metadata = Struct(vec![unknown(), (12, restrictions)]);
        let unread = Struct(vec![
            unknown(),
            (22, metadata),
            (21, Struct(vec![(2, I64(1))])),
        ]);
        let entities = vec![Thrift::generated(classified), unread];
        let bytes = encode_tweet(
            vec![],
            vec![(7, List(TType::Struct, entities)), exclusive_conversation()],
        );

        let features = decode_fixture(&bytes);
        assert_eq!(
            features.media,
            MediaFeature {
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string()],
                geo_deny_list: vec!["de".to_string(), "fr".to_string()],
                ..Default::default()
            }
        );
        assert_eq!(
            features.exclusive_conversation_author_id,
            Some(CONVERSATION_AUTHOR_ID as u64)
        );
    }

    #[test]
    fn keyed_media_without_restrictions_adds_nothing() {
        let keyed = |additional_metadata| MediaEntity {
            media_key: Some(MediaKey::default()),
            additional_metadata,
            ..Default::default()
        };
        let bytes = media_fixture(
            false,
            false,
            None,
            &[
                keyed(None),
                keyed(Some(AdditionalMetadata::default())),
                keyed(Some(AdditionalMetadata {
                    restrictions: Some(Restrictions::default()),
                    ..Default::default()
                })),
            ],
        );

        assert_eq!(decode_fixture(&bytes).media, MediaFeature::default());
    }

    #[test]
    fn media_presence_comes_from_refs_or_card_reference_and_uploads_from_own_media_keys() {
        for (name, bytes, has_media, has_uploaded_media) in [
            (
                "own media keys",
                media_fixture(true, false, Some(true), &[]),
                true,
                true,
            ),
            (
                "empty media keys",
                media_fixture(true, false, Some(false), &[]),
                true,
                false,
            ),
            (
                "no media keys field",
                media_fixture(true, false, None, &[]),
                true,
                false,
            ),
            ("card", media_fixture(false, true, None, &[]), true, false),
            (
                "entities only",
                media_fixture(false, false, None, &[MediaEntity::default()]),
                false,
                false,
            ),
        ] {
            let media = decode_fixture(&bytes).media;
            assert_eq!(
                (media.has_media, media.has_uploaded_media),
                (has_media, has_uploaded_media),
                "{name}"
            );
        }
    }

    #[test]
    fn community_id_is_the_first_of_the_community_id_list() {
        for (community_ids, expected) in [
            (vec![500, 600], NonZeroU64::new(500)),
            (
                vec![1_500_000_000_000_000_000],
                NonZeroU64::new(1_500_000_000_000_000_000),
            ),
            (vec![], None),
        ] {
            let ids = community_ids.iter().map(|&id| I64(id)).collect();
            let communities = Struct(vec![(1, List(TType::I64, ids)), (2, Str("channel".into()))]);
            let bytes = encode_tweet(vec![], vec![(125, communities), exclusive_conversation()]);

            let features = decode_fixture(&bytes);
            assert_eq!(features.community_id, expected, "{community_ids:?}");
            assert_eq!(
                features.exclusive_conversation_author_id,
                Some(CONVERSATION_AUTHOR_ID as u64),
                "{community_ids:?}: the field after communities did not decode"
            );
        }
    }

    #[test]
    fn a_panicking_sub_reader_is_an_error_not_a_panic() {
        let bytes = encode_tweet(vec![], vec![exclusive_conversation()]);
        let end = bytes
            .windows(8)
            .position(|window| window == CONVERSATION_AUTHOR_ID.to_be_bytes())
            .unwrap()
            + 4;

        let error = decode_tweet(&bytes[..end]).unwrap_err();
        assert_eq!(error.to_string(), "MVal decoder panicked");
    }

    #[test]
    fn an_absent_value_is_missing_and_a_value_without_core_data_is_an_error() {
        assert!(decode_tweet(&no_row()).unwrap().is_none());

        let without_core_data = found(Struct(vec![(1, I64(TWEET_ID))]));
        let error = decode_tweet(&without_core_data).unwrap_err();
        assert_eq!(error.to_string(), "tweet value without coreData");
    }

    fn recorded(name: &str) -> Vec<u8> {
        let path = format!("tests/fixtures/tweet_for_visibility/{name}.hex");
        let hex = std::fs::read_to_string(crate::pinned::crate_path(&path)).unwrap();
        hex.trim()
            .as_bytes()
            .chunks(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn recorded_answers_decode_the_features_rules_read() {
        const STALE_EDIT: u64 = 1001;
        const CREATOR_EXCLUSIVE_IMAGE: u64 = 1003;
        const NSFW_ADMIN_IMAGE: u64 = 1004;
        const NSFW_USER_IMAGE: u64 = 1005;
        const LEGACY_MEDIA_POST: u64 = 1006;
        const EXCLUSIVE: u64 = 1007;
        const NSFW_ADMIN_TEXT: u64 = 1008;
        const LINK_CARD: u64 = 1009;
        const EXCLUSIVE_AUTHOR: u64 = 7002;
        let uploaded = || MediaFeature {
            has_media: true,
            has_uploaded_media: true,
            ..Default::default()
        };
        let nsfw = |user, admin| NsfwFeature { user, admin };
        let rows = [
            ("stale_original", STALE_EDIT, TweetFeatures::default()),
            ("stale_edit", STALE_EDIT, TweetFeatures::default()),
            (
                "creator_exclusive_image",
                CREATOR_EXCLUSIVE_IMAGE,
                TweetFeatures {
                    media: uploaded(),
                    exclusive_conversation_author_id: Some(EXCLUSIVE_AUTHOR),
                    ..Default::default()
                },
            ),
            (
                "nsfw_admin_image",
                NSFW_ADMIN_IMAGE,
                TweetFeatures {
                    media: uploaded(),
                    nsfw: nsfw(false, true),
                    ..Default::default()
                },
            ),
            (
                "nsfw_user_image",
                NSFW_USER_IMAGE,
                TweetFeatures {
                    media: uploaded(),
                    nsfw: nsfw(true, false),
                    ..Default::default()
                },
            ),
            (
                "legacy_media_post",
                LEGACY_MEDIA_POST,
                TweetFeatures {
                    media: uploaded(),
                    ..Default::default()
                },
            ),
            (
                "exclusive",
                EXCLUSIVE,
                TweetFeatures {
                    exclusive_conversation_author_id: Some(EXCLUSIVE_AUTHOR),
                    ..Default::default()
                },
            ),
            (
                "nsfw_admin_text",
                NSFW_ADMIN_TEXT,
                TweetFeatures {
                    nsfw: nsfw(false, true),
                    ..Default::default()
                },
            ),
            ("link_card", LINK_CARD, TweetFeatures::default()),
        ];
        for (name, latest_edit, expected) in rows {
            let mut features = decode_fixture(&recorded(name));
            assert_eq!(features.latest_edit_tweet_id, Some(latest_edit), "{name}");
            features.latest_edit_tweet_id = None;
            assert_eq!(features, expected, "{name}");
        }
        assert_eq!(decode_tweet(&recorded("deleted")).unwrap(), None);
    }

    #[test]
    fn the_latest_edit_is_the_last_id_of_the_initial_edit_chain() {
        let initial = |ids: Vec<u64>| EditControlInitial {
            edit_tweet_ids: ids,
            ..Default::default()
        };
        for (control, latest) in [
            (EditControl::Initial(initial(vec![20])), Some(20)),
            (
                EditControl::Edit(EditControlEdit {
                    initial_tweet_id: 10,
                    edit_control_initial: Some(initial(vec![10, 20, 30])),
                }),
                Some(30),
            ),
            (
                EditControl::Edit(EditControlEdit {
                    initial_tweet_id: 10,
                    edit_control_initial: None,
                }),
                None,
            ),
        ] {
            assert_eq!(latest_edit_tweet_id(&control), latest, "{control:?}");
        }
    }
}
