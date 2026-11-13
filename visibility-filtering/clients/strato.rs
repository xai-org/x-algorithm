use anyhow::{bail, Result};
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::{Bytes, MValCodec, StratoGrpc};
use xai_twittercontext_proto::TwitterContextViewer;

const FETCH: &str = "fetch";

pub(crate) async fn fetch(grpc: &StratoGrpc, column: &str, key: Vec<u8>) -> Result<Bytes> {
    grpc.call(column, FETCH, vec![key], None).await
}

pub(crate) async fn fetch_each(
    grpc: &StratoGrpc,
    keys: impl IntoIterator<Item = (&str, Vec<u8>)>,
    viewer_id: Option<u64>,
) -> Vec<Result<Bytes>> {
    let calls = keys
        .into_iter()
        .map(|(column, key)| (column, FETCH, [key]))
        .collect();
    let viewer = viewer_id.map(|user_id| TwitterContextViewer {
        user_id: user_id.cast_signed(),
        ..Default::default()
    });
    grpc.batch_call(calls, viewer.as_ref()).await
}

pub(crate) fn decode<T: TSerializable>(bytes: &[u8]) -> Result<Option<T>> {
    row(strato_decode(bytes)?)
}

pub(crate) fn row<T>(result: StratoResult<T>) -> Result<Option<T>> {
    match result {
        StratoResult::Ok { value, .. } => Ok(value),
        StratoResult::Err { code, message } => bail!("Strato error (code {code}): {message}"),
    }
}

pub(crate) fn decode_only(name: &str) -> thrift::Error {
    thrift::new_protocol_error(
        thrift::ProtocolErrorKind::NotImplemented,
        format!("{name} is decode-only"),
    )
}

pub(crate) trait ToThrift {
    fn write(&self, proto: &mut dyn TOutputProtocol);
}

pub(crate) struct EncodeOnlyCodec<T>(pub(crate) T);

impl<T: ToThrift> MValCodec for EncodeOnlyCodec<T> {
    fn thrift_type() -> TType {
        TType::Struct
    }

    #[expect(clippy::unimplemented, reason = "a key or view is only encoded")]
    fn from_thrift(_proto: &mut dyn TInputProtocol) -> Self {
        unimplemented!("a key or view is only encoded")
    }

    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        self.0.write(proto);
    }
}

#[cfg(test)]
pub(crate) mod answer {
    use thrift::protocol::{
        TBinaryOutputProtocol, TFieldIdentifier, TListIdentifier, TMapIdentifier, TOutputProtocol,
        TSerializable, TSetIdentifier, TStructIdentifier, TType,
    };

    type WriteStruct = Box<dyn Fn(&mut dyn TOutputProtocol)>;

    const MVAL: i16 = 4;
    const OK: i16 = 2556;
    const ERR: i16 = 4421;
    const V: i16 = 118;
    const SOME: i16 = 26900;
    const NONE: i16 = 9048;
    const CODE: i16 = -21011;
    const REASON: i16 = -28092;

    pub(crate) enum Thrift {
        Struct(Vec<(i16, Thrift)>),
        Bool(bool),
        I32(i32),
        I64(i64),
        Str(String),
        List(TType, Vec<Thrift>),
        Set(TType, Vec<Thrift>),
        Map(TType, TType, Vec<(Thrift, Thrift)>),
        Generated(WriteStruct),
        Void,
    }

    impl Thrift {
        pub(crate) fn generated(value: impl TSerializable + 'static) -> Self {
            Self::Generated(Box::new(move |proto| {
                value.write_to_out_protocol(proto).unwrap();
            }))
        }

        pub(crate) fn bytes(&self) -> Vec<u8> {
            let mut bytes = vec![];
            self.write(&mut TBinaryOutputProtocol::new(&mut bytes, false));
            bytes
        }

        fn kind(&self) -> TType {
            match self {
                Self::Struct(_) | Self::Generated(_) => TType::Struct,
                Self::Bool(_) => TType::Bool,
                Self::I32(_) => TType::I32,
                Self::I64(_) => TType::I64,
                Self::Str(_) => TType::String,
                Self::List(..) => TType::List,
                Self::Set(..) => TType::Set,
                Self::Map(..) => TType::Map,
                Self::Void => TType::Void,
            }
        }

        fn write(&self, proto: &mut dyn TOutputProtocol) {
            let len = |len: usize| i32::try_from(len).unwrap();
            match self {
                Self::Struct(fields) => {
                    proto
                        .write_struct_begin(&TStructIdentifier::new(""))
                        .unwrap();
                    for (id, value) in fields {
                        proto
                            .write_field_begin(&TFieldIdentifier::new("", value.kind(), *id))
                            .unwrap();
                        value.write(proto);
                        proto.write_field_end().unwrap();
                    }
                    proto.write_field_stop().unwrap();
                    proto.write_struct_end().unwrap();
                }
                Self::Bool(value) => proto.write_bool(*value).unwrap(),
                Self::I32(value) => proto.write_i32(*value).unwrap(),
                Self::I64(value) => proto.write_i64(*value).unwrap(),
                Self::Str(value) => proto.write_string(value).unwrap(),
                Self::List(kind, values) => {
                    let list = TListIdentifier::new(*kind, len(values.len()));
                    proto.write_list_begin(&list).unwrap();
                    values.iter().for_each(|value| value.write(proto));
                    proto.write_list_end().unwrap();
                }
                Self::Set(kind, values) => {
                    let set = TSetIdentifier::new(*kind, len(values.len()));
                    proto.write_set_begin(&set).unwrap();
                    values.iter().for_each(|value| value.write(proto));
                    proto.write_set_end().unwrap();
                }
                Self::Map(key, value, entries) => {
                    let map = TMapIdentifier::new(*key, *value, len(entries.len()));
                    proto.write_map_begin(&map).unwrap();
                    for (key, value) in entries {
                        key.write(proto);
                        value.write(proto);
                    }
                    proto.write_map_end().unwrap();
                }
                Self::Generated(write) => write(proto),
                Self::Void => {}
            }
        }
    }

    pub(crate) fn found(value: Thrift) -> Vec<u8> {
        ok(Thrift::Struct(vec![(SOME, value)]))
    }

    pub(crate) fn no_row() -> Vec<u8> {
        ok(Thrift::Struct(vec![(NONE, Thrift::Void)]))
    }

    pub(crate) fn strato_error(code: i32, reason: &str) -> Vec<u8> {
        let error = Thrift::Struct(vec![
            (CODE, Thrift::I32(code)),
            (REASON, Thrift::Str(reason.to_owned())),
        ]);
        envelope(ERR, error)
    }

    fn ok(option: Thrift) -> Vec<u8> {
        envelope(OK, Thrift::Struct(vec![(V, option)]))
    }

    fn envelope(arm: i16, value: Thrift) -> Vec<u8> {
        Thrift::Struct(vec![(MVAL, Thrift::Struct(vec![(arm, value)]))]).bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_answer_fails_with_its_code_and_message() {
        let error = row::<()>(StratoResult::Err {
            code: 2,
            message: "boom".to_owned(),
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "Strato error (code 2): boom");
    }
}
