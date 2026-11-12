use anyhow::Result;
use thrift::protocol::TType;
use tonic::async_trait;
use xai_safety_label_source::WarmFetcher;
use xai_strato::strato_thrift::{strato_decode, MValProtocol, StratoRead, StratoResult};
use xai_strato::{encode, Bytes, StratoGrpc};

use crate::clients::strato;
use crate::thrift_read::SkipValue;

const WARM_COLUMN_PATH: &str = "visibility/baseTweetSafetyLabelMap";

pub(crate) struct StratoWarmFetcher(pub(crate) StratoGrpc);

#[async_trait]
impl WarmFetcher for StratoWarmFetcher {
    async fn fetch(&self, ids: &[u64]) -> usize {
        let keys = ids
            .iter()
            .map(|id| (WARM_COLUMN_PATH, encode(&(id.cast_signed(), ()))));
        let answers = strato::fetch_each(&self.0, keys, None).await;
        failed_keys(ids.len(), &answers)
    }
}

fn failed_keys(asked: usize, answers: &[Result<Bytes>]) -> usize {
    let fetched = answers
        .iter()
        .filter(|answer| {
            let reply = answer.as_deref().map(strato_decode::<SkippedLabelMap>);
            matches!(reply, Ok(Ok(StratoResult::Ok { .. })))
        })
        .count();
    asked.saturating_sub(fetched)
}

struct SkippedLabelMap;

impl StratoRead for SkippedLabelMap {
    fn read(proto: &mut MValProtocol<'_>) -> thrift::Result<Self> {
        proto.skip_value(TType::Struct)?;
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::strato::answer::{found, no_row, strato_error, Thrift};
    use anyhow::anyhow;

    #[test]
    fn a_failed_call_a_column_error_and_an_unanswered_key_count_as_failed() {
        let label_map = found(Thrift::Struct(vec![(1, Thrift::Bool(true))]));
        let answered =
            || -> Vec<Result<Bytes>> { vec![Ok(label_map.clone().into()), Ok(no_row().into())] };
        assert_eq!(failed_keys(2, &answered()), 0);
        assert_eq!(failed_keys(3, &answered()), 1);
        let mut failed = answered();
        failed.push(Err(anyhow!("unavailable")));
        failed.push(Ok(strato_error(5, "column failed").into()));
        assert_eq!(failed_keys(4, &failed), 2);
    }
}
