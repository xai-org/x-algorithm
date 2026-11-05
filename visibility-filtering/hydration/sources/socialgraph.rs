use super::{source, Backends, CallContext, Expired, Input, SourceFetch};
use crate::clients::socialgraph_client::EdgeQuery;
use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::hydrator::{Edge, EdgeKey};
use crate::hydration::metrics::record_flock_missing_keys;
use rustc_hash::FxHashSet;
use std::collections::HashSet;
use std::iter;
#[cfg(test)]
use {
    super::InMemoryCall,
    crate::clients::socialgraph_client::{EdgeDirection, Graph},
};

impl SourceFetch for source::Flock {
    const BACKEND_LABEL: &'static str = "socialgraph";
    const DECIDES_EXISTENCE: bool = false;
    const NEEDS_VIEWER: bool = false;
    const EDGE: Option<Edge> = None;
    type Key = EdgeKey;
    type Reply = Vec<RawHydrationBatch<bool>>;

    async fn fetch(backends: &Backends, input: &Input<'_>) -> Result<Self::Reply, Expired> {
        let queries = input.queries;
        let sets = backends
            .socialgraph
            .select_edges(input.viewer_id, queries)
            .await;
        let edges = edge_batches(queries, sets.as_deref());
        select_result(sets.as_deref(), edges)
    }

    fn count_missing(context: &CallContext<'_>, edges: Self::Reply) -> Self::Reply {
        let (edges, missing) = missing_sets_read_no_edge(edges);
        let (backend, call) = context.label;
        record_flock_missing_keys(context.metrics, backend, call, missing);
        edges
    }

    #[cfg(test)]
    type World = FlockWorld;

    #[cfg(test)]
    fn in_memory(world: &FlockWorld, input: &Input<'_>, call: &InMemoryCall<'_>) -> Self::Reply {
        world.select(input, call.fails())
    }
}

fn expired(status: &tonic::Status) -> bool {
    let timer_expired = iter::successors(std::error::Error::source(status), |err| err.source())
        .any(|err| err.is::<tonic::TimeoutExpired>());
    timer_expired || status.code() == tonic::Code::DeadlineExceeded
}

fn select_result(
    sets: Result<&[Option<HashSet<u64>>], &tonic::Status>,
    edges: Vec<RawHydrationBatch<bool>>,
) -> Result<Vec<RawHydrationBatch<bool>>, Expired> {
    match sets {
        Err(status) if expired(status) => Err(Expired),
        Ok(_) | Err(_) => Ok(edges),
    }
}

fn edge_batches(
    queries: &[EdgeQuery],
    sets: Result<&[Option<HashSet<u64>>], &tonic::Status>,
) -> Vec<RawHydrationBatch<bool>> {
    queries
        .iter()
        .enumerate()
        .map(|(position, query)| {
            let set = sets
                .as_ref()
                .map(|sets| sets.get(position).and_then(Option::as_ref));
            let answers = query.destination_ids.iter().map(|&destination| {
                let answer = match set {
                    Err(_) => Hydrated::Failed,
                    Ok(None) => Hydrated::Partial(false),
                    Ok(Some(set)) => Hydrated::Found(set.contains(&destination)),
                };
                (destination, answer)
            });
            HydrationBatch::from_hydrated(answers.collect())
        })
        .collect()
}

fn missing_sets_read_no_edge(
    edges: Vec<RawHydrationBatch<bool>>,
) -> (Vec<RawHydrationBatch<bool>>, usize) {
    let mut missing = FxHashSet::default();
    let edges = edges
        .into_iter()
        .map(|edge| {
            let mut answers = edge.into_hydrated();
            for (destination, answer) in &mut answers {
                if let Hydrated::Partial(holds) = *answer {
                    *answer = Hydrated::Found(holds);
                    missing.insert(*destination);
                }
            }
            HydrationBatch::from_hydrated(answers)
        })
        .collect();
    (edges, missing.len())
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct FlockWorld {
    pub(crate) edges: HashSet<(Graph, u64, u64)>,
    pub(crate) failed_graphs: HashSet<Graph>,
    pub(crate) failed_edges: HashSet<(Graph, u64)>,
    pub(crate) missing_graphs: HashSet<Graph>,
}

#[cfg(test)]
impl FlockWorld {
    fn select(&self, input: &Input<'_>, fails: bool) -> Vec<RawHydrationBatch<bool>> {
        let queries = input.queries;
        let failed_graph = queries.iter().any(|query| {
            self.failed_graphs.contains(&query.graph)
                || query
                    .destination_ids
                    .iter()
                    .any(|&id| self.failed_edges.contains(&(query.graph, id)))
        });
        let answer = |query: &EdgeQuery| {
            let from = query.from.id(input.viewer_id)?;
            let holds = |&id: &u64| {
                let edge = match query.direction {
                    EdgeDirection::Forward => (query.graph, from, id),
                    EdgeDirection::Reverse => (query.graph, id, from),
                };
                self.edges.contains(&edge)
            };
            (!self.missing_graphs.contains(&query.graph)).then(|| {
                query
                    .destination_ids
                    .iter()
                    .copied()
                    .filter(holds)
                    .collect()
            })
        };
        let sets: Result<Vec<_>, _> = (!(fails || failed_graph))
            .then(|| queries.iter().map(answer).collect())
            .ok_or_else(|| tonic::Status::unavailable("select failed"));
        edge_batches(queries, sets.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::Graph;
    use crate::hydration::execute::tests::{executor, VIEWER};
    use crate::hydration::sources::InMemorySources;
    use crate::hydration::Hydrator;
    use crate::rules::SafetyLevel;
    use crate::scenario::raw;
    use tonic::Status;

    #[test]
    fn a_select_past_its_forwarded_deadline_expires_and_another_error_fails_every_key() {
        let queries = [EdgeQuery::forward(Graph::Follows, vec![1, 2])];
        let client_timer = Status::from_error(Box::new(tonic::TimeoutExpired(())));
        for (status, expired) in [
            (client_timer, true),
            (Status::deadline_exceeded(""), true),
            (Status::unavailable(""), false),
        ] {
            let batches = edge_batches(&queries, Err(&status));
            let failed = Some(Hydrated::Failed);
            assert_eq!(
                [1, 2].map(|key| batches[0].hydrated(&key).cloned()),
                [failed.clone(), failed],
                "{status:?}"
            );
            let result = select_result(Err(&status), batches);
            assert_eq!(result.is_err(), expired, "{status:?}");
        }
    }

    #[tokio::test]
    async fn a_query_missing_from_the_select_answer_reads_no_edge_and_fails_no_candidate() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .edge(Graph::Follows, VIEWER, 10)
            .edge(Graph::Mutes, VIEWER, 10)
            .miss_graph(Graph::Mutes);
        let hydrated = executor(sources, SafetyLevel::TimelineHome, Some(VIEWER))
            .hydrate(&[raw(1, None)])
            .await;
        assert_eq!(
            hydrated.candidates[0].edges,
            crate::models::EdgeSet::default().with(Hydrator::Follows)
        );
        assert!(hydrated.failed_ids.is_empty());
    }
}
