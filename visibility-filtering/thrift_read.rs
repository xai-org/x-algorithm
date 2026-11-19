use std::io::{self, BufRead};
use std::ops::Range;
use thrift::protocol::{TFieldIdentifier, TInputProtocol, TType};
use thrift::{ProtocolErrorKind, TConfiguration};
use xai_strato::strato_thrift::MValProtocol;

pub(crate) trait SkipValue: TInputProtocol {
    fn skip_value(&mut self, field_type: TType) -> thrift::Result<()>;
}

impl SkipValue for MValProtocol<'_> {
    fn skip_value(&mut self, field_type: TType) -> thrift::Result<()> {
        skip(self, field_type)
    }
}

impl SkipValue for dyn TInputProtocol + '_ {
    fn skip_value(&mut self, field_type: TType) -> thrift::Result<()> {
        self.skip(field_type)
    }
}

pub(crate) fn read_fields<P: SkipValue + ?Sized>(
    proto: &mut P,
    mut read_field: impl FnMut(&mut P, &TFieldIdentifier) -> thrift::Result<bool>,
) -> thrift::Result<()> {
    proto.read_struct_begin()?;
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        if !read_field(proto, &field)? {
            proto.skip_value(field.field_type)?;
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()
}

pub(crate) const SKIP_DEPTH: i8 = 64;

fn skip(proto: &mut MValProtocol<'_>, field_type: TType) -> thrift::Result<()> {
    skip_till_depth(proto, field_type, SKIP_DEPTH)
}

fn skip_till_depth(
    proto: &mut MValProtocol<'_>,
    field_type: TType,
    depth: i8,
) -> thrift::Result<()> {
    if depth == 0 {
        return Err(thrift::new_protocol_error(
            ProtocolErrorKind::DepthLimit,
            format!("cannot parse past {field_type:?}"),
        ));
    }
    match field_type {
        TType::String => string_range(proto).map(|_| ()),
        TType::Struct => {
            proto.read_struct_begin()?;
            loop {
                let rest = proto.transport.fill_buf()?;
                let field_type = match rest.first() {
                    None => return Err(end_of_buffer()),
                    Some(0) => {
                        proto.transport.consume(1);
                        break;
                    }
                    Some(&byte) => field_type_from_u8(byte)?,
                };
                let value = fixed_width(field_type).filter(|_| depth > 1);
                let len = 3 + value.unwrap_or(0);
                if rest.len() < len {
                    return Err(end_of_buffer());
                }
                proto.transport.consume(len);
                if value.is_none() {
                    skip_till_depth(proto, field_type, depth - 1)?;
                }
            }
            proto.read_struct_end()
        }
        TType::List => {
            let list = proto.read_list_begin()?;
            skip_elements(proto, &[list.element_type], list.size, depth - 1)
        }
        TType::Set => {
            let set = proto.read_set_begin()?;
            skip_elements(proto, &[set.element_type], set.size, depth - 1)
        }
        TType::Map => {
            let map = proto.read_map_begin()?;
            match (map.key_type, map.value_type) {
                (Some(key), Some(value)) => {
                    skip_elements(proto, &[key, value], map.size, depth - 1)
                }
                _ => Ok(()),
            }
        }
        TType::Stop
        | TType::Void
        | TType::Bool
        | TType::I08
        | TType::I16
        | TType::I32
        | TType::I64
        | TType::Double
        | TType::Uuid
        | TType::Utf7 => match fixed_width(field_type) {
            Some(width) => advance(proto, width),
            None => Err(thrift::new_protocol_error(
                ProtocolErrorKind::Unknown,
                format!("cannot skip field type {field_type:?}"),
            )),
        },
    }
}

pub(crate) fn string_range(proto: &mut MValProtocol<'_>) -> thrift::Result<Range<usize>> {
    let size = proto.read_i32()?;
    let len = usize::try_from(size).map_err(|_| {
        thrift::new_protocol_error(
            ProtocolErrorKind::NegativeSize,
            format!("Negative byte array size: {size}"),
        )
    })?;
    if len > TConfiguration::DEFAULT_STRING_LIMIT {
        return Err(thrift::new_protocol_error(
            ProtocolErrorKind::SizeLimit,
            format!(
                "Byte array size {len} exceeds maximum allowed size of {}",
                TConfiguration::DEFAULT_STRING_LIMIT
            ),
        ));
    }
    let start = usize::try_from(proto.transport.position()).map_err(|_| end_of_buffer())?;
    advance(proto, len)?;
    Ok(start..start + len)
}

pub(crate) fn skip_elements(
    proto: &mut MValProtocol<'_>,
    types: &[TType],
    size: i32,
    depth: i8,
) -> thrift::Result<()> {
    let width: Option<usize> = types.iter().map(|&t| fixed_width(t)).sum();
    if let (Some(width), Ok(size)) = (width, usize::try_from(size))
        && depth > 0
    {
        return advance(proto, width * size);
    }
    for _ in 0..size {
        for &element_type in types {
            skip_till_depth(proto, element_type, depth)?;
        }
    }
    Ok(())
}

fn fixed_width(field_type: TType) -> Option<usize> {
    match field_type {
        TType::Bool | TType::I08 => Some(1),
        TType::I16 => Some(2),
        TType::I32 => Some(4),
        TType::I64 | TType::Double => Some(8),
        TType::Uuid => Some(16),
        TType::Stop
        | TType::Void
        | TType::String
        | TType::Struct
        | TType::Map
        | TType::Set
        | TType::List
        | TType::Utf7 => None,
    }
}

fn field_type_from_u8(byte: u8) -> thrift::Result<TType> {
    match byte {
        0x00 => Ok(TType::Stop),
        0x01 => Ok(TType::Void),
        0x02 => Ok(TType::Bool),
        0x03 => Ok(TType::I08),
        0x04 => Ok(TType::Double),
        0x06 => Ok(TType::I16),
        0x08 => Ok(TType::I32),
        0x0a => Ok(TType::I64),
        0x0b => Ok(TType::String),
        0x0c => Ok(TType::Struct),
        0x0d => Ok(TType::Map),
        0x0e => Ok(TType::Set),
        0x0f => Ok(TType::List),
        0x10 => Ok(TType::Uuid),
        _ => Err(thrift::new_protocol_error(
            ProtocolErrorKind::InvalidData,
            format!("cannot convert {byte} to TType"),
        )),
    }
}

fn advance(proto: &mut MValProtocol<'_>, len: usize) -> thrift::Result<()> {
    if proto.transport.fill_buf()?.len() < len {
        return Err(end_of_buffer());
    }
    proto.transport.consume(len);
    Ok(())
}

fn end_of_buffer() -> thrift::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "failed to fill whole buffer").into()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn skip_outcome(
        bytes: &[u8],
        skip_with: impl Fn(&mut MValProtocol<'_>) -> thrift::Result<()>,
    ) -> Result<u64, String> {
        let mut proto = MValProtocol::new(std::io::Cursor::new(bytes), false);
        skip_with(&mut proto)
            .map(|()| proto.transport.position())
            .map_err(|e| format!("{e:?}"))
    }

    pub(crate) fn assert_skips_like_thrift(bytes: &[u8], field_type: TType) {
        assert_eq!(
            skip_outcome(bytes, |proto| skip(proto, field_type)),
            skip_outcome(bytes, |proto| proto.skip(field_type)),
            "{field_type:?} {bytes:02x?}"
        );
    }

    #[test]
    fn skip_steps_over_fixed_width_containers_and_rejects_bad_sizes_as_thrift_does() {
        let size = |n: i32| n.to_be_bytes().to_vec();
        let doubles = [vec![0x04], size(2), vec![7; 16]].concat();
        let uuids = [vec![0x10], size(1), vec![9; 16]].concat();
        let map = [vec![0x06, 0x02], size(3), vec![1; 9]].concat();
        let map_of_lists = [
            vec![0x08, 0x0f],
            size(1),
            size(5),
            vec![0x0a],
            size(1),
            vec![3; 8],
        ]
        .concat();
        for (field_type, bytes) in [
            (TType::List, doubles.clone()),
            (TType::List, doubles[..doubles.len() - 1].to_vec()),
            (TType::Set, uuids),
            (TType::Map, map),
            (TType::Map, map_of_lists),
            (TType::List, [vec![0x01], size(1)].concat()),
            (TType::List, [vec![0x01], size(0)].concat()),
            (TType::List, [vec![0x0a], size(-1)].concat()),
            (TType::List, [vec![0x0a], size(20_000_000)].concat()),
            (TType::List, [vec![0x11], size(0)].concat()),
            (TType::String, size(-1)),
            (TType::String, size(i32::MAX)),
            (TType::String, [size(3), b"ab".to_vec()].concat()),
            (TType::I64, vec![0; 7]),
            (TType::Uuid, vec![0; 16]),
            (TType::Void, vec![]),
            (TType::Stop, vec![]),
        ] {
            assert_skips_like_thrift(&bytes, field_type);
        }
    }

    #[test]
    fn skip_fails_past_the_nesting_thrift_allows_as_thrift_does() {
        let i64s = [vec![0x0a, 0, 0, 0, 2], vec![5; 16]].concat();
        for depth in [1, 62, 63, 64, 65, 70] {
            let structs = [
                [0x0c, 0, 1].repeat(depth),
                vec![0x0f, 0, 1],
                i64s.clone(),
                vec![0; depth + 1],
            ]
            .concat();
            let lists = [[0x0f, 0, 0, 0, 1].repeat(depth), i64s.clone()].concat();
            let scalar = |field: &[u8]| {
                [
                    [0x0c, 0, 1].repeat(depth),
                    field.to_vec(),
                    vec![0; depth + 1],
                ]
                .concat()
            };
            let i64_field = scalar(&[0x0a, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9]);
            let bool_field = scalar(&[0x02, 0, 1, 1]);
            assert_skips_like_thrift(&structs, TType::Struct);
            assert_skips_like_thrift(&lists, TType::List);
            assert_skips_like_thrift(&i64_field, TType::Struct);
            assert_skips_like_thrift(&bool_field, TType::Struct);
            for (bytes, field_type) in [
                (&structs, TType::Struct),
                (&lists, TType::List),
                (&i64_field, TType::Struct),
                (&bool_field, TType::Struct),
            ] {
                let outcome = skip_outcome(bytes, |proto| skip(proto, field_type));
                match depth {
                    1 => assert_eq!(outcome, Ok(bytes.len() as u64)),
                    70 => assert!(outcome.unwrap_err().contains("DepthLimit")),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn skip_reads_struct_field_headers_as_thrift_does_on_every_prefix() {
        let field = |field_type: u8, value: &[u8]| [&[field_type, 0x7f, 0xfe][..], value].concat();
        let every_fixed_width = [
            field(0x02, &[0x00]),
            field(0x02, &[0x01]),
            field(0x02, &[0xff]),
            field(0x03, &[0x80]),
            field(0x04, &[1; 8]),
            field(0x06, &[1; 2]),
            field(0x08, &[1; 4]),
            field(0x0a, &[1; 8]),
            field(0x10, &[1; 16]),
            vec![0x00, 0xaa],
        ]
        .concat();
        let nested = [
            field(0x0c, &[0x00]),
            field(0x0c, &field(0x0a, &[2; 8])),
            vec![0x00],
            field(0x0b, &[0, 0, 0, 2, b'h', b'i']),
            vec![0x0a, 0xff, 0xff],
            vec![3; 8],
            field(0x0f, &[0x0c, 0, 0, 0, 2, 0x00, 0x02, 0, 1, 1, 0x00]),
            vec![0x00],
        ]
        .concat();
        let mut cases = vec![every_fixed_width, nested, field(0x0a, &[1; 8])];
        for byte in [0x01, 0x05, 0x07, 0x09, 0x11, 0x7f, 0x80, 0xff] {
            cases.push(vec![byte, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            cases.push([field(0x08, &[1; 4]), vec![byte, 0, 1, 0]].concat());
        }
        for bytes in cases {
            for end in 0..=bytes.len() {
                assert_skips_like_thrift(&bytes[..end], TType::Struct);
            }
        }
    }
}
