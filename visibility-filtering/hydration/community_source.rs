use crate::clients::strato::{self, decode_only};
use crate::models::CommunityModeration;
use crate::thrift_read::read_fields;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TOutputProtocol, TSerializable, TStructIdentifier, TType,
};
use xai_strato::{encode, Bytes, MValCodec, StratoGrpc};

const TWEET_MODERATION: &str = "communities/moderationState.TweetCommunityRelationship";
const AUTHOR_MODERATION: &str = "communities/moderationState.UserCommunityRelationship";
const VISIBILITY_FEATURES: &str = "communities/visibility/visibilityFeatures.Community";
const IS_REMOVED: &str = "communities/isRemoved.Community";

const HIDDEN: i16 = 2;
const REMOVED: i16 = 2;
const V1: i16 = 1;
const VIEWER_IS_COMMUNITY_MODERATOR: i16 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CommunityPost {
    pub(crate) tweet_id: u64,
    pub(crate) author_id: u64,
    pub(crate) community_id: u64,
}

pub(crate) struct CommunitySource {
    pub(crate) grpc_client: StratoGrpc,
}

impl CommunitySource {
    pub(crate) async fn moderations(
        &self,
        posts: &[CommunityPost],
    ) -> HashMap<u64, Result<Option<CommunityModeration>>> {
        let keys = posts.iter().flat_map(|post| {
            let relationship = UserCommunityRelationshipId {
                user_id: post.author_id,
                community_id: post.community_id,
            };
            [
                (TWEET_MODERATION, encode(&(post.tweet_id, ()))),
                (AUTHOR_MODERATION, encode(&(relationship, ()))),
            ]
        });
        let mut replies = strato::fetch_each(&self.grpc_client, keys, None)
            .await
            .into_iter();
        posts
            .iter()
            .map(|post| {
                let is_hidden = read(replies.next(), |arm: UnionArm| arm.0 == Some(HIDDEN));
                let is_author_removed =
                    read(replies.next(), |arm: UnionArm| arm.0 == Some(REMOVED));
                let moderation = is_hidden.and_then(|is_hidden| {
                    Ok(Some(CommunityModeration {
                        is_hidden,
                        is_author_removed: is_author_removed?,
                    }))
                });
                (post.tweet_id, moderation)
            })
            .collect()
    }

    pub(crate) async fn moderators(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> HashMap<u64, Result<Option<bool>>> {
        self.fetch_as_viewer(
            VISIBILITY_FEATURES,
            viewer_id,
            community_ids,
            |features: ViewerIsModerator| features.0,
        )
        .await
    }

    pub(crate) async fn viewer_removals(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> HashMap<u64, Result<Option<bool>>> {
        self.fetch_as_viewer(
            IS_REMOVED,
            viewer_id,
            community_ids,
            |removed: IsRemoved| removed.0,
        )
        .await
    }

    async fn fetch_as_viewer<T: TSerializable>(
        &self,
        column: &str,
        viewer_id: u64,
        community_ids: &[u64],
        holds: impl Fn(T) -> bool,
    ) -> HashMap<u64, Result<Option<bool>>> {
        let keys = community_ids
            .iter()
            .map(|&community_id| (column, encode(&(community_id, ()))));
        let mut replies = strato::fetch_each(&self.grpc_client, keys, Some(viewer_id))
            .await
            .into_iter();
        community_ids
            .iter()
            .map(|&community_id| (community_id, read(replies.next(), &holds).map(Some)))
            .collect()
    }
}

fn read<T: TSerializable>(
    reply: Option<Result<Bytes>>,
    holds: impl FnOnce(T) -> bool,
) -> Result<bool> {
    let bytes = reply.ok_or_else(|| anyhow!("Strato answered fewer calls than it was sent"))??;
    Ok(strato::decode::<T>(&bytes)?.is_some_and(holds))
}

#[derive(Default)]
struct UserCommunityRelationshipId {
    user_id: u64,
    community_id: u64,
}

impl UserCommunityRelationshipId {
    fn write(&self, proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        proto.write_struct_begin(&TStructIdentifier::new("UserCommunityRelationshipId"))?;
        for (id, value) in [(1, self.user_id), (2, self.community_id)] {
            proto.write_field_begin(&TFieldIdentifier::new("", TType::I64, id))?;
            proto.write_i64(value.cast_signed())?;
            proto.write_field_end()?;
        }
        proto.write_field_stop()?;
        proto.write_struct_end()
    }

    fn read(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut key = Self::default();
        read_fields(proto, |proto, field| {
            if is(field, 1, TType::I64) {
                key.user_id = proto.read_i64()?.cast_unsigned();
            } else if is(field, 2, TType::I64) {
                key.community_id = proto.read_i64()?.cast_unsigned();
            } else {
                return Ok(false);
            }
            Ok(true)
        })?;
        Ok(key)
    }
}

impl MValCodec for UserCommunityRelationshipId {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        Self::read(proto).unwrap_or_default()
    }

    #[expect(
        clippy::expect_used,
        reason = "the binary protocol writes into the encode buffer, which cannot fail"
    )]
    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        self.write(proto)
            .expect("UserCommunityRelationshipId encodes");
    }
}

struct UnionArm(Option<i16>);

impl TSerializable for UnionArm {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut arm = None;
        read_fields(proto, |_, field| {
            arm = arm.or(field.id);
            Ok(false)
        })?;
        Ok(Self(arm))
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(decode_only("UnionArm"))
    }
}

struct ViewerIsModerator(bool);

impl TSerializable for ViewerIsModerator {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut is_moderator = false;
        read_fields(proto, |proto, field| {
            if !is(field, V1, TType::Struct) {
                return Ok(false);
            }
            read_fields(proto, |proto, field| {
                if !is(field, VIEWER_IS_COMMUNITY_MODERATOR, TType::Bool) {
                    return Ok(false);
                }
                is_moderator = proto.read_bool()?;
                Ok(true)
            })?;
            Ok(true)
        })?;
        Ok(Self(is_moderator))
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(decode_only("ViewerIsModerator"))
    }
}

struct IsRemoved(bool);

impl TSerializable for IsRemoved {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        Ok(Self(proto.read_bool()?))
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(decode_only("IsRemoved"))
    }
}

fn is(field: &TFieldIdentifier, id: i16, field_type: TType) -> bool {
    field.id == Some(id) && field.field_type == field_type
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::Thrift::{Bool, Struct, I32};
    use crate::clients::strato::answer::{found, no_row, strato_error};
    use thrift::protocol::{TBinaryInputProtocol, TBinaryOutputProtocol};

    fn answered(bytes: Vec<u8>) -> Option<Result<Bytes>> {
        Some(Ok(Bytes::from(bytes)))
    }

    #[test]
    fn a_moderation_state_reads_its_union_arm_and_an_error_answer_fails() {
        let hidden = |arm: UnionArm| arm.0 == Some(HIDDEN);
        let state = |arm| answered(found(Struct(vec![(arm, Struct(vec![]))])));
        let rows = [
            (state(2), Some(true)),
            (state(1), Some(false)),
            (answered(no_row()), Some(false)),
            (answered(strato_error(5, "denied")), None),
            (Some(Err(anyhow!("Strato error (code 5): denied"))), None),
            (Some(Ok(Bytes::from_static(&[0xFF]))), None),
            (None, None),
        ];
        for (reply, expected) in rows {
            assert_eq!(read(reply, hidden).ok(), expected);
        }
    }

    #[test]
    fn the_moderator_flag_is_read_from_the_v1_arm() {
        for flag in [true, false] {
            let v1 = Struct(vec![(1, I32(1)), (4, Bool(flag))]);
            let features = answered(found(Struct(vec![(1, v1)])));
            let is_moderator = read(features, |features: ViewerIsModerator| features.0);
            assert_eq!(is_moderator.ok(), Some(flag));
        }
    }

    #[test]
    fn the_removal_flag_is_the_column_bool_and_no_row_is_not_removed() {
        for (reply, expected) in [
            (answered(found(Bool(true))), Some(true)),
            (answered(found(Bool(false))), Some(false)),
            (answered(no_row()), Some(false)),
            (answered(strato_error(5, "denied")), None),
        ] {
            let is_removed = read(reply, |removed: IsRemoved| removed.0);
            assert_eq!(is_removed.ok(), expected);
        }
    }

    #[test]
    fn the_author_moderation_key_writes_its_thrift_field_ids() {
        let key = UserCommunityRelationshipId {
            user_id: 1_500_000_000_000_000_000,
            community_id: 7,
        };
        let mut bytes = Vec::new();
        key.write(&mut TBinaryOutputProtocol::new(&mut bytes, true))
            .unwrap();
        assert_eq!(
            bytes,
            [
                &[0x0A, 0x00, 0x01][..],
                &1_500_000_000_000_000_000_i64.to_be_bytes(),
                &[0x0A, 0x00, 0x02],
                &7_i64.to_be_bytes(),
                &[0x00],
            ]
            .concat()
        );
        let read = UserCommunityRelationshipId::read(&mut TBinaryInputProtocol::new(
            bytes.as_slice(),
            true,
        ))
        .unwrap();
        assert_eq!(
            (read.user_id, read.community_id),
            (key.user_id, key.community_id)
        );
    }
}
