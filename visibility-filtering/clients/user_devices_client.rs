use crate::clients::strato::{self, EncodeOnlyCodec, ToThrift};
use crate::thrift_read::read_fields;
use std::collections::HashSet;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TListIdentifier, TOutputProtocol, TStructIdentifier, TType,
};
use xai_strato::strato_thrift::{strato_decode, MValProtocol, StratoRead};
use xai_strato::{encode, StratoGrpc};

pub(crate) const CONFIRMED_EMAILS_COLUMN: &str = "notifinfra/service/getObfuscatedUserEmails";
pub(crate) const VERIFIED_PHONES_COLUMN: &str = "notifinfra/service/getObfuscatedUserPhones";

const CORE_SERVICE: i32 = 12;

pub async fn confirmed_emails(grpc: &StratoGrpc, user_ids: &[u64]) -> anyhow::Result<HashSet<u64>> {
    let bytes = strato::fetch(grpc, CONFIRMED_EMAILS_COLUMN, emails_key(user_ids)).await?;
    Ok(strato::row(strato_decode::<ConfirmedEmails>(&bytes)?)?
        .map_or_else(HashSet::new, |emails| emails.0))
}

pub async fn verified_phones(grpc: &StratoGrpc, user_ids: &[u64]) -> anyhow::Result<HashSet<u64>> {
    let bytes = strato::fetch(grpc, VERIFIED_PHONES_COLUMN, phones_key(user_ids)).await?;
    Ok(strato::row(strato_decode::<VerifiedPhones>(&bytes)?)?
        .map_or_else(HashSet::new, |phones| phones.0))
}

const EMAIL_PURPOSES_FIELD: i16 = 5;
const PHONE_PURPOSES_FIELD: i16 = 3;

pub(crate) fn emails_key(user_ids: &[u64]) -> Vec<u8> {
    users_key(user_ids, EMAIL_PURPOSES_FIELD)
}

pub(crate) fn phones_key(user_ids: &[u64]) -> Vec<u8> {
    users_key(user_ids, PHONE_PURPOSES_FIELD)
}

fn users_key(user_ids: &[u64], purposes_field: i16) -> Vec<u8> {
    encode(&(
        EncodeOnlyCodec(UsersRequest {
            user_ids,
            purposes_field,
        }),
        (),
    ))
}

struct UsersRequest<'a> {
    user_ids: &'a [u64],
    purposes_field: i16,
}

impl ToThrift for UsersRequest<'_> {
    #[expect(
        clippy::unwrap_used,
        reason = "`xai_strato::encode` writes into a Vec, which cannot fail; 2^31 ids would not fit in memory first"
    )]
    fn write(&self, proto: &mut dyn TOutputProtocol) {
        proto
            .write_struct_begin(&TStructIdentifier::new("UsersRequest"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("userIds", TType::List, 1))
            .unwrap();
        let size = i32::try_from(self.user_ids.len()).unwrap();
        proto
            .write_list_begin(&TListIdentifier::new(TType::I64, size))
            .unwrap();
        for &user_id in self.user_ids {
            proto.write_i64(user_id.cast_signed()).unwrap();
        }
        proto.write_list_end().unwrap();
        proto.write_field_end().unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new(
                "purposesOfProcessing",
                TType::List,
                self.purposes_field,
            ))
            .unwrap();
        proto
            .write_list_begin(&TListIdentifier::new(TType::I32, 1))
            .unwrap();
        proto.write_i32(CORE_SERVICE).unwrap();
        proto.write_list_end().unwrap();
        proto.write_field_end().unwrap();
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }
}

fn read_users(
    i_prot: &mut MValProtocol<'_>,
    mut read_user: impl FnMut(&mut MValProtocol<'_>) -> thrift::Result<Option<u64>>,
) -> thrift::Result<HashSet<u64>> {
    let mut users = HashSet::new();
    read_fields(i_prot, |i_prot, field| {
        if field.id != Some(1) || field.field_type != TType::List {
            return Ok(false);
        }
        let list = i_prot.read_list_begin()?;
        for _ in 0..list.size {
            users.extend(read_user(i_prot)?);
        }
        i_prot.read_list_end()?;
        Ok(true)
    })?;
    Ok(users)
}

struct ConfirmedEmails(HashSet<u64>);

impl StratoRead for ConfirmedEmails {
    fn read(i_prot: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        read_users(i_prot, |i_prot| {
            let mut user_id = None;
            let mut confirmed = false;
            read_fields(i_prot, |i_prot, field| match (field.id, field.field_type) {
                (Some(1), TType::I64) => {
                    user_id = Some(i_prot.read_i64()?.cast_unsigned());
                    Ok(true)
                }
                (Some(2), TType::Struct) => {
                    confirmed = true;
                    Ok(false)
                }
                _ => Ok(false),
            })?;
            Ok(user_id.filter(|_| confirmed))
        })
        .map(Self)
    }
}

struct VerifiedPhones(HashSet<u64>);

impl StratoRead for VerifiedPhones {
    fn read(i_prot: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        read_users(i_prot, |i_prot| {
            let mut user_id = None;
            read_fields(i_prot, |i_prot, field| {
                if field.id == Some(2) && field.field_type == TType::I64 {
                    user_id = Some(i_prot.read_i64()?.cast_unsigned());
                    return Ok(true);
                }
                Ok(false)
            })?;
            Ok(user_id)
        })
        .map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::Thrift;
    use std::io::Cursor;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn email_and_phone_requests_carry_core_service() {
        assert_eq!(
            hex(&emails_key(&[7, 8])),
            "0c00040c00000f00010a00000002000000000000000700000000000000080f00050800000001\
             0000000c000100010000"
        );
        assert_eq!(
            hex(&phones_key(&[7])),
            "0c00040c00000f00010a0000000100000000000000070f000308000000010000000c\
             000100010000"
        );
    }

    #[test]
    fn a_confirmed_email_or_any_phone_names_its_user() {
        use Thrift::{List, Str, Struct, I64};
        let emails = Struct(vec![(
            1,
            List(
                TType::Struct,
                vec![
                    Struct(vec![(1, I64(7)), (2, Struct(vec![(1, Str("a".into()))]))]),
                    Struct(vec![(1, I64(8)), (3, Struct(vec![(1, Str("b".into()))]))]),
                ],
            ),
        )]);
        let bytes = emails.bytes();
        let decoded =
            ConfirmedEmails::read(&mut MValProtocol::new(Cursor::new(&bytes[..]), false)).unwrap();
        assert_eq!(decoded.0, HashSet::from([7]));

        let phones = Struct(vec![(
            1,
            List(
                TType::Struct,
                vec![Struct(vec![(1, Str("+1".into())), (2, I64(9))])],
            ),
        )]);
        let bytes = phones.bytes();
        let decoded =
            VerifiedPhones::read(&mut MValProtocol::new(Cursor::new(&bytes[..]), false)).unwrap();
        assert_eq!(decoded.0, HashSet::from([9]));
    }
}
