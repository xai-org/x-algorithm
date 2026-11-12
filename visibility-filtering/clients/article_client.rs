use crate::clients::strato;
use crate::thrift_read::read_fields;
use std::collections::HashMap;
use thrift::protocol::{TInputProtocol, TSerializable, TType};
use xai_strato::strato_thrift::{
    strato_decode, MValProtocol, StratoRead, StratoResult, AUTHORIZATION_ERROR,
};
use xai_strato::StratoGrpc;
use xai_x_thrift::api_media::ApiMediaKey;

const LIFECYCLE_COLUMN: &str = "article/fields/lifecycleState.ArticleEntity";
const MEDIA_ENTITIES_COLUMN: &str = "article/fields/mediaEntities.ArticleEntity";
const COVER_MEDIA_COLUMN: &str = "article/fields/coverMedia.ArticleEntity";

pub async fn lifecycles(
    grpc: &StratoGrpc,
    article_ids: &[u64],
) -> HashMap<u64, anyhow::Result<Option<i32>>> {
    fetch_articles(grpc, None, article_ids, LIFECYCLE_COLUMN, decode_lifecycle).await
}

pub async fn media_entities(
    grpc: &StratoGrpc,
    viewer_id: Option<u64>,
    article_ids: &[u64],
) -> HashMap<u64, anyhow::Result<Option<bool>>> {
    fetch_articles(
        grpc,
        viewer_id,
        article_ids,
        MEDIA_ENTITIES_COLUMN,
        decode_media_entities,
    )
    .await
}

pub async fn cover_media(
    grpc: &StratoGrpc,
    viewer_id: Option<u64>,
    article_ids: &[u64],
) -> HashMap<u64, anyhow::Result<Option<bool>>> {
    fetch_articles(
        grpc,
        viewer_id,
        article_ids,
        COVER_MEDIA_COLUMN,
        decode_cover_media,
    )
    .await
}

async fn fetch_articles<T>(
    grpc: &StratoGrpc,
    viewer_id: Option<u64>,
    article_ids: &[u64],
    column: &str,
    decode: fn(&[u8]) -> anyhow::Result<Option<T>>,
) -> HashMap<u64, anyhow::Result<Option<T>>> {
    let keys = article_ids
        .iter()
        .map(|id| (column, xai_strato::encode(&(id.cast_signed(), ()))));
    let replies = strato::fetch_each(grpc, keys, viewer_id).await;
    article_ids
        .iter()
        .copied()
        .zip(replies.into_iter().map(|reply| decode(&reply?)))
        .collect()
}

fn decode_lifecycle(bytes: &[u8]) -> anyhow::Result<Option<i32>> {
    Ok(strato::row(strato_decode::<ArticleLifecycleState>(bytes)?)?.map(|state| state.lifecycle))
}

fn decode_media_entities(bytes: &[u8]) -> anyhow::Result<Option<bool>> {
    Ok(decode_unless_refused::<MediaEntitiesPresence>(bytes)?.map(|presence| presence.0))
}

fn decode_cover_media(bytes: &[u8]) -> anyhow::Result<Option<bool>> {
    Ok(decode_unless_refused::<ApiMediaKey>(bytes)?.map(|_| true))
}

fn decode_unless_refused<T: StratoRead>(bytes: &[u8]) -> anyhow::Result<Option<T>> {
    let result = strato_decode::<T>(bytes)?;
    if matches!(
        result,
        StratoResult::Err {
            code: AUTHORIZATION_ERROR,
            ..
        }
    ) {
        return Ok(None);
    }
    strato::row(result)
}

struct MediaEntitiesPresence(bool);

impl StratoRead for MediaEntitiesPresence {
    fn read(proto: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        let set = proto.read_set_begin()?;
        if set.element_type != TType::Struct || set.size < 0 {
            return Err(thrift::new_protocol_error(
                thrift::ProtocolErrorKind::InvalidData,
                "article media entities is not a set of ApiMediaKey",
            ));
        }
        for _ in 0..set.size {
            ApiMediaKey::read_from_in_protocol(proto)?;
        }
        proto.read_set_end()?;
        Ok(Self(set.size > 0))
    }
}

struct ArticleLifecycleState {
    lifecycle: i32,
}

const LIFECYCLE: i16 = 1;

impl StratoRead for ArticleLifecycleState {
    fn read(i_prot: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        let mut lifecycle = None;
        read_fields(i_prot, |i_prot, field| {
            if field.id != Some(LIFECYCLE) || field.field_type != TType::I32 {
                return Ok(false);
            }
            lifecycle = Some(i_prot.read_i32()?);
            Ok(true)
        })?;
        let Some(lifecycle) = lifecycle else {
            return Err(thrift::new_protocol_error(
                thrift::ProtocolErrorKind::InvalidData,
                "ArticleLifecycleState has no lifecycle",
            ));
        };
        Ok(Self { lifecycle })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::Thrift::{Struct, I32, I64};
    use std::io::Cursor;

    #[test]
    fn media_presence_reads_sets_and_keys_without_retaining_their_payloads() {
        use crate::clients::strato::answer::{found, no_row, Thrift};
        use Thrift::Set;
        let key = || Struct(vec![(1, Struct(vec![(1, I64(7)), (2, I64(0))]))]);
        assert_eq!(
            decode_media_entities(&found(Set(TType::Struct, vec![key(), key()]))).unwrap(),
            Some(true)
        );
        assert_eq!(
            decode_media_entities(&found(Set(TType::Struct, vec![]))).unwrap(),
            Some(false)
        );
        assert_eq!(decode_media_entities(&no_row()).unwrap(), None);
        assert_eq!(decode_cover_media(&found(key())).unwrap(), Some(true));
        assert_eq!(decode_cover_media(&no_row()).unwrap(), None);
        assert_eq!(
            decode_cover_media(&found(Struct(vec![(4, Struct(vec![]))]))).unwrap(),
            Some(true)
        );
    }

    #[test]
    fn media_errors_and_malformed_keys_are_not_missing_rows() {
        use crate::clients::strato::answer::{found, strato_error, Thrift};
        use Thrift::Set;
        for decode in [decode_media_entities, decode_cover_media] {
            assert!(decode(&strato_error(5, "failed fetch")).is_err());
            assert!(decode(&strato_error(400, "Missing ArticleMetadata for 1")).is_err());
            assert!(decode(&[0]).is_err());
        }
        for fields in [
            vec![],
            vec![(99, Struct(vec![]))],
            vec![(1, Struct(vec![])), (2, Struct(vec![]))],
        ] {
            assert!(decode_cover_media(&found(Struct(fields))).is_err());
        }
        assert!(decode_media_entities(&found(Set(TType::Struct, vec![Struct(vec![])]))).is_err());
        assert!(decode_media_entities(&found(Set(TType::I64, vec![I64(7)]))).is_err());
        let mut truncated = found(Set(
            TType::Struct,
            vec![Struct(vec![(1, Struct(vec![(1, I64(7)), (2, I64(0))]))])],
        ));
        truncated.truncate(25);
        assert!(decode_media_entities(&truncated).is_err());
    }

    #[test]
    fn an_article_the_viewer_may_not_read_has_no_media() {
        use crate::clients::strato::answer::strato_error;
        for decode in [decode_media_entities, decode_cover_media] {
            for reason in [
                "Only subscribers have access",
                "Only article author has access",
            ] {
                assert_eq!(decode(&strato_error(202, reason)).unwrap(), None);
            }
        }
    }

    fn decode_row(lifecycle: Option<i32>) -> thrift::Result<i32> {
        let lifecycle = lifecycle.map(|lifecycle| (LIFECYCLE, I32(lifecycle)));
        let fields = lifecycle.into_iter().chain([(2, I64(1_790_813_967))]);
        let bytes = Struct(fields.collect()).bytes();
        let mut i_prot = MValProtocol::new(Cursor::new(&bytes[..]), false);
        ArticleLifecycleState::read(&mut i_prot).map(|state| state.lifecycle)
    }

    #[test]
    fn reads_the_lifecycle_field_and_rejects_a_row_without_one() {
        assert_eq!(decode_row(Some(4)).unwrap(), 4);
        assert!(decode_row(None).is_err());
    }
}
