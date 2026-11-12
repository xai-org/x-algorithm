use crate::clients::strato;
use crate::thrift_read::read_fields;
use thrift::protocol::{TInputProtocol, TType};
use xai_strato::strato_thrift::{strato_decode, MValProtocol, StratoRead};
use xai_strato::StratoGrpc;

const TFE_TOP_COUNTRY_COLUMN: &str = "about_this_account/tfe_top_country.User";

pub async fn tfe_top_country(grpc: &StratoGrpc, user_id: u64) -> anyhow::Result<Option<String>> {
    let key = xai_strato::encode(&(user_id.cast_signed(), ()));
    let bytes = strato::fetch(grpc, TFE_TOP_COUNTRY_COLUMN, key).await?;
    Ok(strato::row(strato_decode::<UserTfeTopCountry>(&bytes)?)?
        .and_then(|row| row.weighted_top_country))
}

#[derive(Debug, Default, PartialEq)]
struct UserTfeTopCountry {
    weighted_top_country: Option<String>,
}

const WEIGHTED_TOP_COUNTRY: i16 = 2;
const COUNTRY_ID: i16 = 1;

impl StratoRead for UserTfeTopCountry {
    fn read(i_prot: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        let mut weighted_top_country = None;
        read_fields(i_prot, |i_prot, field| {
            if field.id != Some(WEIGHTED_TOP_COUNTRY) || field.field_type != TType::Struct {
                return Ok(false);
            }
            read_fields(i_prot, |i_prot, field| {
                if field.id != Some(COUNTRY_ID) || field.field_type != TType::String {
                    return Ok(false);
                }
                weighted_top_country = Some(i_prot.read_string()?);
                Ok(true)
            })?;
            Ok(true)
        })?;
        Ok(Self {
            weighted_top_country,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::Thrift::{Bool, Str, Struct, I64};
    use std::io::Cursor;

    fn decode_row(details: &[(i16, &str)]) -> UserTfeTopCountry {
        let details = details.iter().map(|&(id, country_id)| {
            let country = Struct(vec![(1, Str(country_id.to_owned())), (2, Bool(true))]);
            (id, country)
        });
        let fields = [(1, I64(7))]
            .into_iter()
            .chain(details)
            .chain([(5, I64(1))]);
        let bytes = Struct(fields.collect()).bytes();
        UserTfeTopCountry::read(&mut MValProtocol::new(Cursor::new(&bytes[..]), false)).unwrap()
    }

    #[test]
    fn decodes_only_the_weighted_top_country() {
        let full = decode_row(&[(3, "FR"), (2, "nz"), (4, "DE")]);
        assert_eq!(full.weighted_top_country.as_deref(), Some("nz"));
        assert_eq!(decode_row(&[(3, "FR")]), UserTfeTopCountry::default());
    }
}
