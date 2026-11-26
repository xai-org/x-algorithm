use crate::clients::strato::{self, EncodeOnlyCodec, ToThrift};
use crate::thrift_read::read_fields;
use std::collections::HashSet;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TListIdentifier, TOutputProtocol, TSetIdentifier,
    TStructIdentifier, TType,
};
use xai_strato::strato_thrift::{strato_decode, MValProtocol, StratoRead};
use xai_strato::{encode, StratoGrpc};

const USER_LOCATION_COLUMN: &str = "geo/service/userLocation";

const PLACE_TYPES: [i32; 5] = [4, 3, 7, 9, 1];

const GEODUCK_IDL_DEFAULT_FLAGS: [(&str, i16, bool); 3] = [
    ("simple_reverse_geocode", 3, false),
    ("include_debug_information", 4, false),
    ("filter_by_confidence", 5, true),
];

const FOUND: i16 = 1;
const PLACE_MAP: i16 = 5;

pub async fn places(grpc: &StratoGrpc, user_id: u64) -> anyhow::Result<HashSet<u64>> {
    let bytes = strato::fetch(grpc, USER_LOCATION_COLUMN, key(user_id)).await?;
    decode_places(&bytes, user_id)
}

pub(crate) fn key(user_id: u64) -> Vec<u8> {
    let view = UserLocationRequest {
        user_id: user_id.cast_signed(),
    };
    encode(&((), EncodeOnlyCodec(view)))
}

fn decode_places(bytes: &[u8], user_id: u64) -> anyhow::Result<HashSet<u64>> {
    Ok(strato::row(strato_decode::<UserLocationResponse>(bytes)?)?
        .and_then(|response| {
            response
                .found
                .into_iter()
                .find(|(user, _)| *user == user_id)
        })
        .map(|(_, places)| places)
        .unwrap_or_default())
}

struct UserLocationRequest {
    user_id: i64,
}

impl ToThrift for UserLocationRequest {
    #[expect(
        clippy::unwrap_used,
        reason = "`xai_strato::encode` writes into a Vec, which cannot fail, and five place types fit an i32"
    )]
    fn write(&self, proto: &mut dyn TOutputProtocol) {
        proto
            .write_struct_begin(&TStructIdentifier::new("UserLocationRequest"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("userIds", TType::List, 1))
            .unwrap();
        proto
            .write_list_begin(&TListIdentifier::new(TType::I64, 1))
            .unwrap();
        proto.write_i64(self.user_id).unwrap();
        proto.write_list_end().unwrap();
        proto.write_field_end().unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("place_query", TType::Struct, 2))
            .unwrap();
        proto
            .write_struct_begin(&TStructIdentifier::new("PlaceQuery"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("place_types", TType::Set, 1))
            .unwrap();
        let place_types = i32::try_from(PLACE_TYPES.len()).unwrap();
        proto
            .write_set_begin(&TSetIdentifier::new(TType::I32, place_types))
            .unwrap();
        for place_type in PLACE_TYPES {
            proto.write_i32(place_type).unwrap();
        }
        proto.write_set_end().unwrap();
        proto.write_field_end().unwrap();
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
        proto.write_field_end().unwrap();
        for (name, id, value) in GEODUCK_IDL_DEFAULT_FLAGS {
            proto
                .write_field_begin(&TFieldIdentifier::new(name, TType::Bool, id))
                .unwrap();
            proto.write_bool(value).unwrap();
            proto.write_field_end().unwrap();
        }
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }
}

struct UserLocationResponse {
    found: Vec<(u64, HashSet<u64>)>,
}

impl StratoRead for UserLocationResponse {
    fn read(i_prot: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        let mut found = Vec::new();
        read_fields(i_prot, |i_prot, field| {
            if field.id != Some(FOUND) || field.field_type != TType::Map {
                return Ok(false);
            }
            let users = i_prot.read_map_begin()?;
            for _ in 0..users.size {
                let user = i_prot.read_i64()?.cast_unsigned();
                found.push((user, read_place_ids(i_prot)?));
            }
            i_prot.read_map_end()?;
            Ok(true)
        })?;
        Ok(Self { found })
    }
}

fn read_place_ids(i_prot: &mut MValProtocol<'_>) -> thrift::Result<HashSet<u64>> {
    let mut places = HashSet::new();
    read_fields(i_prot, |i_prot, field| {
        if field.id != Some(PLACE_MAP) || field.field_type != TType::Map {
            return Ok(false);
        }
        let place_map = i_prot.read_map_begin()?;
        for _ in 0..place_map.size {
            let is_asked = PLACE_TYPES.contains(&i_prot.read_i32()?);
            let ids = i_prot.read_set_begin()?;
            for _ in 0..ids.size {
                let id = i_prot.read_i64()?.cast_unsigned();
                if is_asked {
                    places.insert(id);
                }
            }
            i_prot.read_set_end()?;
        }
        i_prot.read_map_end()?;
        Ok(true)
    })?;
    Ok(places)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::Thrift::{Map, Set, Struct, I32, I64};
    use crate::clients::strato::answer::{found, no_row};

    const USER: u64 = 7;
    const OTHER_USER: u64 = 8;
    const CITY: i32 = 3;
    const COUNTRY: i32 = 1;
    const METRO: i32 = 7;
    const POI: i32 = 5;
    const ABOVE_I64_MAX: u64 = 0xa000_0000_0000_0001;

    type Found<'a> = (u64, Option<&'a [(i32, u64)]>);

    fn located(users: &[Found<'_>]) -> Vec<u8> {
        let id = |id: u64| I64(id.cast_signed());
        let users = users.iter().map(|&(user, place_map)| {
            let place_map = place_map.map(|place_map| {
                let places = place_map.iter().map(|&(place_type, place)| {
                    (I32(place_type), Set(TType::I64, vec![id(place)]))
                });
                (5, Map(TType::I32, TType::Set, places.collect()))
            });
            let fields = [(1, id(user))]
                .into_iter()
                .chain(place_map)
                .chain([(4, I64(1))]);
            (id(user), Struct(fields.collect()))
        });
        found(Struct(vec![(
            1,
            Map(TType::I64, TType::Struct, users.collect()),
        )]))
    }

    #[test]
    fn the_view_asks_the_five_place_types_of_one_user() {
        assert_eq!(
            key(7),
            [
                0x0c, 0x00, 0x04, 0x01, 0x00, 0x00, 0x0c, 0x00, 0x01, 0x0f, 0x00, 0x01, 0x0a, 0x00,
                0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x0c, 0x00, 0x02,
                0x0e, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
                0x00, 0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x02, 0x00, 0x03, 0x00, 0x02, 0x00, 0x04, 0x00, 0x02, 0x00, 0x05, 0x01, 0x00,
                0x00, 0x00,
            ]
        );
    }

    #[test]
    fn decodes_the_users_ids_of_the_five_place_types() {
        let response = located(&[
            (OTHER_USER, Some(&[(CITY, 21)])),
            (
                USER,
                Some(&[(CITY, 11), (COUNTRY, 12), (POI, 13), (METRO, ABOVE_I64_MAX)]),
            ),
        ]);

        assert_eq!(
            decode_places(&response, USER).unwrap(),
            HashSet::from([11, 12, ABOVE_I64_MAX])
        );
    }

    #[test]
    fn a_user_geoduck_does_not_locate_has_no_places() {
        for (name, response) in [
            (
                "only another user found",
                located(&[(OTHER_USER, Some(&[(CITY, 11)]))]),
            ),
            ("found without a place map", located(&[(USER, None)])),
            ("no value", no_row()),
        ] {
            assert_eq!(
                decode_places(&response, USER).unwrap(),
                HashSet::new(),
                "{name}"
            );
        }
    }
}
