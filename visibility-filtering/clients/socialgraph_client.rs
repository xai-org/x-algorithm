use anyhow::Context;
use std::collections::HashSet;
use tonic::async_trait;
use tonic::metadata::MetadataMap;
use tracing::warn;
use xai_flock_client::{FlockClient, FlockClientConfig, FlockTlsConfig};
use xai_flock_proto::{
    EdgeState, LongList, Page, QueryTerm, Results, SelectOperation, SelectOperationType,
    SelectQuery, SelectRequest,
};
use xai_x_rpc::balanced_channel::LbPolicy;

const REVERSE_EDGE_CHUNK_SIZE: usize = 500;

const FLOCK_APERTURE_SIZE: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
pub enum Graph {
    Follows = 1,
    Blocks = 3,
    MuteRetweets = 10,
    Mutes = 23,
    SuperFollows = 55,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum EdgeDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeFrom {
    Viewer,
    User(u64),
}

impl EdgeFrom {
    pub fn id(self, viewer_id: Option<u64>) -> Option<u64> {
        match self {
            EdgeFrom::Viewer => viewer_id,
            EdgeFrom::User(id) => Some(id),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeQuery {
    pub graph: Graph,
    pub direction: EdgeDirection,
    pub from: EdgeFrom,
    pub destination_ids: Vec<u64>,
}

#[cfg(test)]
impl EdgeQuery {
    pub fn forward(graph: Graph, destination_ids: Vec<u64>) -> Self {
        Self {
            graph,
            direction: EdgeDirection::Forward,
            from: EdgeFrom::Viewer,
            destination_ids,
        }
    }

    pub fn reverse(graph: Graph, destination_ids: Vec<u64>) -> Self {
        Self {
            graph,
            direction: EdgeDirection::Reverse,
            from: EdgeFrom::Viewer,
            destination_ids,
        }
    }
}

#[async_trait]
pub trait SocialgraphClient: Send + Sync {
    async fn select_edges(
        &self,
        viewer_id: Option<u64>,
        queries: &[EdgeQuery],
    ) -> Result<Vec<Option<HashSet<u64>>>, tonic::Status>;
}

fn decode_packed_ids(packed: &[u8]) -> impl Iterator<Item = u64> + '_ {
    let (chunks, _remainder) = packed.as_chunks::<8>();
    chunks
        .iter()
        .map(|&arr| i64::from_le_bytes(arr).cast_unsigned())
}

fn edge_membership_query(
    source_id: u64,
    graph: Graph,
    direction: EdgeDirection,
    destination_ids: &[u64],
) -> SelectQuery {
    SelectQuery {
        operations: vec![SelectOperation {
            operation_type: SelectOperationType::SimpleQuery as i32,
            term: Some(QueryTerm {
                source_id: source_id.cast_signed(),
                graph_id: graph as i32,
                is_forward: direction == EdgeDirection::Forward,
                destination_ids: Some(LongList {
                    ids: destination_ids.iter().map(|&id| id.cast_signed()).collect(),
                }),
                state_ids: vec![EdgeState::Positive as i32],
                size_hint: None,
                cursor_hint: None,
            }),
        }],
        page: Some(Page {
            count: i32::try_from(destination_ids.len()).unwrap_or(i32::MAX),
            cursor: -1,
        }),
    }
}

fn select_request(
    viewer_id: Option<u64>,
    queries: &[EdgeQuery],
) -> Result<(SelectRequest, Vec<usize>), tonic::Status> {
    let mut flock_queries = Vec::new();
    let mut counts = Vec::with_capacity(queries.len());
    for query in queries {
        let source_id = query
            .from
            .id(viewer_id)
            .ok_or_else(|| tonic::Status::invalid_argument("an edge query names no source"))?;
        let ids = &query.destination_ids;
        let chunk_size = match query.direction {
            EdgeDirection::Forward => ids.len().max(1),
            EdgeDirection::Reverse => REVERSE_EDGE_CHUNK_SIZE,
        };
        let before = flock_queries.len();
        flock_queries
            .extend(ids.chunks(chunk_size).map(|chunk| {
                edge_membership_query(source_id, query.graph, query.direction, chunk)
            }));
        counts.push(flock_queries.len() - before);
    }
    let request = SelectRequest {
        queries: flock_queries,
        ancestor_client_id: None,
        service_account: None,
        quota_name: None,
    };
    Ok((request, counts))
}

fn decode_edge_sets(counts: &[usize], results: Vec<Results>) -> Vec<Option<HashSet<u64>>> {
    let mut results = results.into_iter();
    counts
        .iter()
        .map(|&count| {
            let mut set = HashSet::new();
            let mut answered = 0;
            for chunk in results.by_ref().take(count) {
                answered += 1;
                set.extend(decode_packed_ids(&chunk.ids));
            }
            (answered == count).then_some(set)
        })
        .collect()
}

pub struct ProdSocialgraphClient {
    flock_client: FlockClient,
    metadata: Option<MetadataMap>,
}

impl ProdSocialgraphClient {
    pub async fn new(
        datacenter: &str,
        ca_cert_path: &str,
        client_cert_path: &str,
        client_key_path: &str,
        deterministic_aperture: bool,
        metadata: Option<MetadataMap>,
    ) -> anyhow::Result<Self> {
        let wilyns = xai_wily::WilyNs::new(xai_wily::WilyConfig {
            zone: datacenter.to_string(),
            ..Default::default()
        })
        .context("Failed to create WilyNS for Flock")?;
        let config = FlockClientConfig {
            tls: Some(FlockTlsConfig::from_s2s(
                ca_cert_path,
                client_cert_path,
                client_key_path,
                datacenter,
            )),
            num_endpoints: usize::MAX,
            aperture_size: Some(FLOCK_APERTURE_SIZE),
            deterministic_aperture,
            lb_policy: Some(LbPolicy::least_request()),
            ..Default::default()
        };
        let flock_client = FlockClient::with_config(wilyns, config).await?;
        Ok(Self {
            flock_client,
            metadata,
        })
    }
}

#[async_trait]
impl SocialgraphClient for ProdSocialgraphClient {
    async fn select_edges(
        &self,
        viewer_id: Option<u64>,
        queries: &[EdgeQuery],
    ) -> Result<Vec<Option<HashSet<u64>>>, tonic::Status> {
        let (request, counts) = select_request(viewer_id, queries)?;
        let mut request = tonic::Request::new(request);
        if let Some(metadata) = &self.metadata {
            *request.metadata_mut() = metadata.clone();
        }
        xai_x_rpc::apply_call_deadline(&mut request);
        match self.flock_client.inner().clone().select(request).await {
            Ok(resp) => Ok(decode_edge_sets(&counts, resp.into_inner().results)),
            Err(e) => {
                let graphs: Vec<Graph> = queries.iter().map(|query| query.graph).collect();
                warn!(error = %e, ?graphs, "FlockDB select failed");
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(ids: &[i64]) -> Results {
        Results {
            ids: ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
            next_cursor: 0,
            prev_cursor: 0,
        }
    }

    #[test]
    fn decode_packed_ids_decodes_little_endian_chunks_and_ignores_trailing_bytes() {
        let packed = [
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa,
            0x99, 0x88, 0x42, 0x24,
        ];
        assert_eq!(
            decode_packed_ids(&packed).collect::<Vec<_>>(),
            [0x0102_0304_0506_0708, 0x8899_aabb_ccdd_eeff]
        );
        assert_eq!(decode_packed_ids(&[]).next(), None);
    }

    #[test]
    fn select_request_sends_forward_queries_whole_and_reverse_queries_in_chunks() {
        let reverse_ids: Vec<u64> = (1..=REVERSE_EDGE_CHUNK_SIZE as u64 + 1).collect();
        let (request, counts) = select_request(
            Some(999),
            &[
                EdgeQuery::forward(Graph::Follows, reverse_ids.clone()),
                EdgeQuery::forward(Graph::Mutes, vec![]),
                EdgeQuery::reverse(Graph::Blocks, reverse_ids.clone()),
                EdgeQuery::reverse(Graph::Follows, vec![]),
            ],
        )
        .unwrap();
        assert_eq!(counts, vec![1, 0, 2, 0]);
        let terms: Vec<(i32, bool, Vec<i64>)> = request
            .queries
            .iter()
            .map(|query| {
                let term = query.operations[0].term.as_ref().unwrap();
                assert_eq!(term.source_id, 999);
                let ids = term.destination_ids.as_ref().unwrap().ids.clone();
                assert_eq!(query.page.unwrap().count, ids.len() as i32);
                (term.graph_id, term.is_forward, ids)
            })
            .collect();
        let ids: Vec<i64> = reverse_ids.iter().map(|&id| id.cast_signed()).collect();
        let (first, second) = ids.split_at(REVERSE_EDGE_CHUNK_SIZE);
        assert_eq!(
            terms,
            vec![
                (1, true, ids.clone()),
                (3, false, first.to_vec()),
                (3, false, second.to_vec()),
            ]
        );
    }

    #[test]
    fn a_sourced_query_sends_its_source_and_needs_no_viewer() {
        let sourced = |source_id, graph, destination_ids| EdgeQuery {
            from: EdgeFrom::User(source_id),
            ..EdgeQuery::forward(graph, destination_ids)
        };
        let (request, _) = select_request(
            None,
            &[
                sourced(20, Graph::Blocks, vec![30, 31]),
                sourced(30, Graph::Follows, vec![20]),
            ],
        )
        .unwrap();
        let terms: Vec<(i64, i32, Vec<i64>)> = request
            .queries
            .iter()
            .map(|query| {
                let term = query.operations[0].term.as_ref().unwrap();
                let ids = term.destination_ids.as_ref().unwrap().ids.clone();
                (term.source_id, term.graph_id, ids)
            })
            .collect();
        assert_eq!(terms, vec![(20, 3, vec![30, 31]), (30, 1, vec![20])]);
        let unsourced = select_request(None, &[EdgeQuery::forward(Graph::Follows, vec![20])]);
        assert_eq!(unsourced.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn decode_edge_sets_unions_each_querys_chunks_and_marks_missing_slots() {
        assert_eq!(
            decode_edge_sets(&[2, 0, 1], vec![pack(&[1]), pack(&[2]), pack(&[3])]),
            vec![
                Some(HashSet::from([1, 2])),
                Some(HashSet::new()),
                Some(HashSet::from([3]))
            ]
        );
        assert_eq!(
            decode_edge_sets(&[1, 2], vec![pack(&[4]), pack(&[5])]),
            vec![Some(HashSet::from([4])), None]
        );
    }
}
