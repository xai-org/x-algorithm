use super::{CallRequest, FetcherMut, FetcherRef, Store};
use crate::hydration::batch::{HydrationBatch, RawHydrationBatch};
use crate::hydration::fetcher::{AnyFetcher, Fetcher};
use crate::hydration::hydrator::{Edge, EdgeKey, UserPair};
use crate::hydration::inventory::Answer;
use crate::hydration::sources::Batches;
use crate::hydration::Hydrator;

impl Store {
    pub(super) fn claim_viewer_edge(&mut self, node: Hydrator, keys: Vec<u64>) -> Vec<u64> {
        let edges: Vec<EdgeKey> = keys
            .into_iter()
            .filter_map(|key| self.viewer_edge(node, key))
            .collect();
        match self.edge_fetcher_mut(node) {
            Some(fetcher) => fetcher
                .claim(edges)
                .into_iter()
                .map(|edge| edge.pair.destination)
                .collect(),
            None => Vec::new(),
        }
    }

    fn viewer_edge(&self, node: Hydrator, destination: u64) -> Option<EdgeKey> {
        let pair = UserPair {
            source: self.facts.viewer_id()?,
            destination,
        };
        edge_key(node, pair)
    }

    pub(super) fn edge_fetcher(&self, node: Hydrator) -> Option<&Fetcher<bool, EdgeKey>> {
        match self.fetchers.get(self.spec(node).fetcher()) {
            FetcherRef::Edges(fetcher) => Some(fetcher),
            FetcherRef::Keyed(_) => None,
        }
    }

    pub(super) fn edge_fetcher_mut(
        &mut self,
        node: Hydrator,
    ) -> Option<&mut Fetcher<bool, EdgeKey>> {
        let fetcher = self.spec(node).fetcher();
        match self.fetchers.get_mut(fetcher) {
            FetcherMut::Edges(fetcher) => Some(fetcher),
            FetcherMut::Keyed(_) => None,
        }
    }

    pub(super) fn answer(&self, node: Hydrator, key: u64) -> Option<Answer> {
        if node.is_edge() {
            return Some(
                self.edge_fetcher(node)?
                    .answer(self.viewer_edge(node, key)?),
            );
        }
        Some(self.fetcher(node)?.answer(key))
    }

    pub(super) fn edge_holds(&self, node: Hydrator, key: u64) -> Option<bool> {
        self.edge_fetcher(node)?
            .get(self.viewer_edge(node, key)?)
            .copied()
    }
}

pub(super) fn edge_key(node: Hydrator, pair: UserPair) -> Option<EdgeKey> {
    Some(EdgeKey {
        edge: node.edge()?,
        pair,
    })
}

pub(in crate::hydration) trait Lands<K>: Batches {
    fn land(
        self,
        fetcher: &mut Fetcher<Self::Value, K>,
        call: &CallRequest<'_>,
        edge: Option<Edge>,
    );
}

impl<V: Send> Lands<u64> for RawHydrationBatch<V> {
    fn land(self, fetcher: &mut Fetcher<V>, call: &CallRequest<'_>, _: Option<Edge>) {
        fetcher.land(&call.keys, self);
    }
}

impl Lands<EdgeKey> for RawHydrationBatch<bool> {
    fn land(
        self,
        fetcher: &mut Fetcher<bool, EdgeKey>,
        call: &CallRequest<'_>,
        edge: Option<Edge>,
    ) {
        if let (Some(edge), Some(source)) = (edge, call.viewer_id) {
            land_edge(fetcher, edge, source, &call.keys, self);
        }
    }
}

impl Lands<EdgeKey> for Vec<RawHydrationBatch<bool>> {
    fn land(self, fetcher: &mut Fetcher<bool, EdgeKey>, call: &CallRequest<'_>, _: Option<Edge>) {
        debug_assert_eq!(self.len(), call.queries.len(), "one answer per query");
        for ((edge, query), answer) in call.edge_queries().zip(self) {
            if let Some(source) = query.from.id(call.viewer_id) {
                land_edge(fetcher, edge, source, &query.destination_ids, answer);
            }
        }
    }
}

fn land_edge(
    fetcher: &mut Fetcher<bool, EdgeKey>,
    edge: Edge,
    source: u64,
    destinations: &[u64],
    answers: RawHydrationBatch<bool>,
) {
    let key = |destination| EdgeKey {
        edge,
        pair: UserPair {
            source,
            destination,
        },
    };
    let claimed: Vec<EdgeKey> = destinations.iter().copied().map(key).collect();
    let answers = answers
        .into_hydrated()
        .into_iter()
        .map(|(destination, holds)| (key(destination), holds))
        .collect();
    fetcher.land(&claimed, HydrationBatch::from_hydrated(answers));
}
