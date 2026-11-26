use crate::clients::strato::{self, EncodeOnlyCodec, ToThrift};
use anyhow::Result;
use std::collections::HashMap;
use std::iter;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TOutputProtocol, TStructIdentifier, TType,
};
use xai_strato::strato_thrift::{strato_decode, MValProtocol, StratoRead};
use xai_strato::{encode, StratoGrpc};

const IS_MEMBER_COLUMN: &str = "trusted-friends/visibility-filtering/isMember.TrustedFriendsList";
const IS_OWNER_COLUMN: &str = "trusted-friends/visibility-filtering/isOwner.TrustedFriendsList";

pub async fn batch_is_member_or_owner(
    grpc: &StratoGrpc,
    viewer_id: u64,
    list_ids: &[u64],
) -> HashMap<u64, Result<bool>> {
    let keys = list_ids.iter().flat_map(|&list_id| {
        let key = key(list_id, viewer_id);
        [(IS_MEMBER_COLUMN, key.clone()), (IS_OWNER_COLUMN, key)]
    });
    let mut answers = strato::fetch_each(grpc, keys, None)
        .await
        .into_iter()
        .map(|bytes| decode_answer(&bytes?));
    let lists = iter::from_fn(|| Some(member_or_owner(answers.next()?, answers.next()?)));
    list_ids.iter().copied().zip(lists).collect()
}

pub(crate) fn key(list_id: u64, viewer_id: u64) -> Vec<u8> {
    let view = View {
        user_id: viewer_id.cast_signed(),
    };
    encode(&(list_id.cast_signed(), EncodeOnlyCodec(view)))
}

fn member_or_owner(member: Result<bool>, owner: Result<bool>) -> Result<bool> {
    match (member, owner) {
        (Ok(true), _) | (_, Ok(true)) => Ok(true),
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(false), Ok(false)) => Ok(false),
    }
}

fn decode_answer(bytes: &[u8]) -> Result<bool> {
    Ok(strato::row(strato_decode::<Answer>(bytes)?)?.is_some_and(|Answer(answer)| answer))
}

struct View {
    user_id: i64,
}

const USER_ID_FIELD: i16 = 11846;

impl ToThrift for View {
    #[expect(
        clippy::unwrap_used,
        reason = "`xai_strato::encode` writes into a Vec, which cannot fail"
    )]
    fn write(&self, proto: &mut dyn TOutputProtocol) {
        proto
            .write_struct_begin(&TStructIdentifier::new("View"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("userId", TType::I64, USER_ID_FIELD))
            .unwrap();
        proto.write_i64(self.user_id).unwrap();
        proto.write_field_end().unwrap();
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }
}

struct Answer(bool);

impl StratoRead for Answer {
    fn read(i_prot: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        Ok(Self(i_prot.read_bool()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::{found, no_row, Thrift};
    use anyhow::anyhow;

    fn answer(value: Option<bool>) -> bool {
        let bytes = value.map_or_else(no_row, |value| found(Thrift::Bool(value)));
        decode_answer(&bytes).unwrap()
    }

    #[test]
    fn the_key_names_the_list_and_the_viewer_as_userid() {
        assert_eq!(
            key(9001, 50),
            [
                0x0c, 0x00, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x29,
                0x0c, 0x00, 0x01, 0x0a, 0x2e, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x32,
                0x00, 0x00, 0x00,
            ]
        );
    }

    #[test]
    fn the_column_answer_decodes_and_a_missing_value_answers_false() {
        assert!(answer(Some(true)));
        assert!(!answer(Some(false)));
        assert!(!answer(None));
    }

    #[test]
    fn either_column_answering_true_shows_and_otherwise_a_failure_fails() {
        let failed = || Err(anyhow!("unavailable"));
        assert!(member_or_owner(Ok(true), failed()).unwrap());
        assert!(member_or_owner(failed(), Ok(true)).unwrap());
        assert!(!member_or_owner(Ok(false), Ok(false)).unwrap());
        assert!(member_or_owner(Ok(false), failed()).is_err());
        assert!(member_or_owner(failed(), Ok(false)).is_err());
    }
}
