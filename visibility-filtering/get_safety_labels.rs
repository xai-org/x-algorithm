use crate::metrics::families::{LOOKUP_FAILURES, LOOKUP_TWEET_IDS};
use crate::metrics::Metrics;
use enum_map::EnumMap;
use prost::bytes::BufMut;
use prost::encoding::{hash_map, message, uint64};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::codec::{Codec, EncodeBuf, Encoder};
use tonic::{Request, Response, Status};
use tonic_prost::ProstDecoder;
use xai_safety_label_source::{FailureKind, LookupError, SafetyLabelSource};
use xai_visibility_filtering_proto as vf_pb;

pub struct GetSafetyLabelsEndpoint {
    safety_label_source: Arc<SafetyLabelSource>,
    metrics: Metrics,
}

impl GetSafetyLabelsEndpoint {
    pub(crate) fn new(safety_label_source: Arc<SafetyLabelSource>, metrics: Metrics) -> Self {
        Self {
            safety_label_source,
            metrics,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<Response<CachedGetSafetyLabelsResponse>, Status> {
        crate::metrics::observe_request(
            &self.metrics,
            crate::metrics::Endpoint::GetSafetyLabels,
            crate::metrics::RequestLevel::Absent,
            request,
            |request, _, request_metrics| self.handle_inner(request, request_metrics),
        )
        .await
    }

    async fn handle_inner(
        &self,
        request: Request<vf_pb::GetSafetyLabelsRequest>,
        request_metrics: crate::metrics::RequestMetrics,
    ) -> Result<Response<CachedGetSafetyLabelsResponse>, Status> {
        let req = request.into_inner();

        let unique_ids: Vec<u64> = req
            .tweet_ids
            .into_iter()
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect();
        crate::metrics::record_batch_size(&request_metrics, unique_ids.len());
        let metrics = request_metrics.plain();

        let resolved = self.safety_label_source.get(&unique_ids).await;
        let (response, failure_counts) =
            CachedGetSafetyLabelsResponse::try_from_resolved(resolved, unique_ids.len())?;

        metrics.count(
            &LOOKUP_TWEET_IDS,
            ["success"],
            response.results.len() as u64,
        );
        metrics.count(
            &LOOKUP_TWEET_IDS,
            ["failed"],
            response.failed_ids.len() as u64,
        );
        for (kind, &count) in &failure_counts {
            metrics.count(&LOOKUP_FAILURES, [kind.into()], count as u64);
        }

        Ok(Response::new(response))
    }
}

#[derive(Debug)]
pub(crate) struct CachedGetSafetyLabelsResponse {
    results: HashMap<u64, Arc<vf_pb::SafetyLabelMap>>,
    failed_ids: Vec<u64>,
}

impl CachedGetSafetyLabelsResponse {
    fn try_from_resolved(
        resolved: FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
        requested_count: usize,
    ) -> Result<(Self, EnumMap<FailureKind, usize>), Status> {
        let mut results = HashMap::with_capacity(resolved.len());
        let mut failed_ids = Vec::new();
        let mut failures = EnumMap::default();

        for (id, lookup_result) in resolved {
            match lookup_result {
                Ok(label_map) => {
                    results.insert(id, label_map);
                }
                Err(e) => {
                    failed_ids.push(id);
                    failures[e.kind] += 1;
                }
            }
        }

        failed_ids.sort_unstable();

        let accounted_count = results.len() + failed_ids.len();
        if accounted_count != requested_count {
            return Err(Status::internal(format!(
                "GetSafetyLabels invariant violation: requested={requested_count} accounted={accounted_count}"
            )));
        }

        Ok((
            Self {
                results,
                failed_ids,
            },
            failures,
        ))
    }

    fn encode(&self, buf: &mut impl BufMut) {
        hash_map::encode(
            uint64::encode,
            uint64::encoded_len,
            |tag, label_map: &Arc<vf_pb::SafetyLabelMap>, buf| {
                message::encode(tag, &**label_map, buf);
            },
            |tag, label_map: &Arc<vf_pb::SafetyLabelMap>| message::encoded_len(tag, &**label_map),
            1,
            &self.results,
            buf,
        );
        uint64::encode_packed(2, &self.failed_ids, buf);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GetSafetyLabelsCodec;

impl Codec for GetSafetyLabelsCodec {
    type Encode = CachedGetSafetyLabelsResponse;
    type Decode = vf_pb::GetSafetyLabelsRequest;
    type Encoder = Self;
    type Decoder = ProstDecoder<vf_pb::GetSafetyLabelsRequest>;

    fn encoder(&mut self) -> Self {
        *self
    }

    fn decoder(&mut self) -> Self::Decoder {
        ProstDecoder::default()
    }
}

impl Encoder for GetSafetyLabelsCodec {
    type Item = CachedGetSafetyLabelsResponse;
    type Error = Status;

    fn encode(
        &mut self,
        response: CachedGetSafetyLabelsResponse,
        buf: &mut EncodeBuf<'_>,
    ) -> Result<(), Status> {
        response.encode(buf);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tonic::Code;

    fn labels_with_entry(type_id: i32) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                type_id,
                vf_pb::SafetyLabel {
                    score: Some(0.5),
                    applicable_users: Vec::new(),
                    holdback_experiment: None,
                    source: None,
                    created_at_msec: None,
                    expires_at_msec: None,
                    applicable_countries: Vec::new(),
                    safety_label_source: None,
                },
            )]),
        }
    }

    #[test]
    fn try_from_resolved_reports_partial_failure() {
        let (response, failure_counts) = CachedGetSafetyLabelsResponse::try_from_resolved(
            FxHashMap::from_iter([
                (1, Ok(Arc::new(labels_with_entry(11)))),
                (4, Ok(Arc::new(labels_with_entry(44)))),
                (
                    2,
                    Err(LookupError::new(FailureKind::ManhattanDecode, "decode")),
                ),
                (
                    3,
                    Err(LookupError::new(FailureKind::ManhattanFetch, "mh down")),
                ),
            ]),
            4,
        )
        .unwrap();

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.failed_ids, vec![2, 3]);
        assert_eq!(failure_counts[FailureKind::ManhattanFetch], 1);
        assert_eq!(failure_counts[FailureKind::ManhattanDecode], 1);
    }

    fn prod_shaped_label_map(id: u64) -> vf_pb::SafetyLabelMap {
        let label_count = 3 + i32::try_from(id % 4).unwrap();
        vf_pb::SafetyLabelMap {
            labels: (0..label_count)
                .map(|i| {
                    let msec = (1_i64 << 40) + i64::from(i);
                    let label = vf_pb::SafetyLabel {
                        score: Some(f64::from(i) / 7.0),
                        applicable_users: Vec::new(),
                        holdback_experiment: None,
                        source: Some(format!("botmaker_rule_source_{id:04}_{i:02}")),
                        created_at_msec: Some(msec),
                        expires_at_msec: Some(msec + (1 << 30)),
                        applicable_countries: if i % 3 == 0 {
                            vec!["us".to_owned(), "de".to_owned()]
                        } else {
                            Vec::new()
                        },
                        safety_label_source: Some(
                            vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                                vf_pb::BotmakerAction { rule_id: msec },
                            ),
                        ),
                    };
                    (i * 7 + 1, label)
                })
                .collect(),
        }
    }

    #[test]
    fn cached_response_encodes_the_generated_bytes() {
        let hasher = std::hash::RandomState::new();
        let cases = [
            ("empty", Vec::new(), Vec::new()),
            ("failed only", Vec::new(), vec![3, 1 << 40]),
            (
                "empty label map",
                vec![(5, vf_pb::SafetyLabelMap::default())],
                Vec::new(),
            ),
            ("id zero", vec![(0, prod_shaped_label_map(0))], Vec::new()),
            ("single", vec![(7, prod_shaped_label_map(7))], vec![8]),
            (
                "multi",
                (1..=50)
                    .map(|id| (id, prod_shaped_label_map(id)))
                    .chain([(51, vf_pb::SafetyLabelMap::default())])
                    .collect(),
                vec![52, 53],
            ),
        ];
        for (name, label_maps, failed_ids) in cases {
            let mut generated = HashMap::with_hasher(hasher.clone());
            let mut cached = HashMap::with_hasher(hasher.clone());
            for (id, label_map) in label_maps {
                cached.insert(id, Arc::new(label_map.clone()));
                generated.insert(id, label_map);
            }
            let generated_bytes = vf_pb::GetSafetyLabelsResponse {
                results: generated,
                failed_ids: failed_ids.clone(),
            }
            .encode_to_vec();
            let mut cached_bytes = Vec::new();
            CachedGetSafetyLabelsResponse {
                results: cached,
                failed_ids,
            }
            .encode(&mut cached_bytes);
            assert_eq!(cached_bytes, generated_bytes, "{name}");
        }
    }

    #[test]
    fn try_from_resolved_rejects_missing_ids() {
        let status = CachedGetSafetyLabelsResponse::try_from_resolved(
            FxHashMap::from_iter([(7, Ok(Arc::new(labels_with_entry(22))))]),
            2,
        )
        .unwrap_err();

        assert_eq!(status.code(), Code::Internal);
        assert!(status.message().contains("requested=2 accounted=1"));
    }
}
