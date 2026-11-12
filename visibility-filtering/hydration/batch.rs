use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::slice;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Hydrated<V> {
    Found(V),
    Partial(V),
    NotFound,
    Failed,
    Unanswered,
}

impl<V> Hydrated<V> {
    pub(crate) fn value(&self) -> Option<&V> {
        match self {
            Hydrated::Found(value) | Hydrated::Partial(value) => Some(value),
            Hydrated::NotFound | Hydrated::Failed | Hydrated::Unanswered => None,
        }
    }

    pub(crate) fn into_value(self) -> Option<V> {
        match self {
            Hydrated::Found(value) | Hydrated::Partial(value) => Some(value),
            Hydrated::NotFound | Hydrated::Failed | Hydrated::Unanswered => None,
        }
    }

    pub(crate) fn is_complete(&self) -> bool {
        matches!(self, Hydrated::Found(_) | Hydrated::NotFound)
    }

    pub(crate) fn map<V2>(self, f: impl FnOnce(V) -> V2) -> Hydrated<V2> {
        match self {
            Hydrated::Found(value) => Hydrated::Found(f(value)),
            Hydrated::Partial(value) => Hydrated::Partial(f(value)),
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed => Hydrated::Failed,
            Hydrated::Unanswered => Hydrated::Unanswered,
        }
    }
}

impl<V, E> From<Result<Option<V>, E>> for Hydrated<V> {
    fn from(result: Result<Option<V>, E>) -> Self {
        match result {
            Ok(Some(value)) => Hydrated::Found(value),
            Ok(None) => Hydrated::NotFound,
            Err(_) => Hydrated::Failed,
        }
    }
}

#[derive(Clone)]
pub(crate) struct HydrationBatch<K, V> {
    results: FxHashMap<K, Hydrated<V>>,
}

pub(crate) type RawHydrationBatch<V> = HydrationBatch<u64, V>;

impl<K: Eq + Hash, V> HydrationBatch<K, V> {
    pub(crate) fn from_results<E, S: BuildHasher>(
        expected: impl IntoIterator<Item = K>,
        results: HashMap<K, Result<Option<V>, E>, S>,
    ) -> Self {
        Self::from_results_with(expected, results, Hydrated::from)
    }

    pub(crate) fn from_results_with<R, S: BuildHasher>(
        expected: impl IntoIterator<Item = K>,
        mut results: HashMap<K, R, S>,
        mut decode: impl FnMut(R) -> Hydrated<V>,
    ) -> Self {
        Self::from_expected(expected, |key| {
            results
                .remove(key)
                .map_or(Hydrated::Unanswered, &mut decode)
        })
    }

    pub(crate) fn from_hydrated(results: FxHashMap<K, Hydrated<V>>) -> Self {
        Self { results }
    }

    fn from_expected(
        expected: impl IntoIterator<Item = K>,
        mut resolve: impl FnMut(&K) -> Hydrated<V>,
    ) -> Self {
        let expected = expected.into_iter();
        let mut results =
            FxHashMap::with_capacity_and_hasher(expected.size_hint().0, Default::default());
        for key in expected {
            if let Entry::Vacant(entry) = results.entry(key) {
                let hydrated = resolve(entry.key());
                entry.insert(hydrated);
            }
        }
        Self { results }
    }

    pub(crate) fn failed(expected: impl IntoIterator<Item = K>) -> Self {
        Self {
            results: expected
                .into_iter()
                .map(|key| (key, Hydrated::Failed))
                .collect(),
        }
    }

    pub(crate) fn hydrated(&self, key: &K) -> Option<&Hydrated<V>> {
        self.results.get(key)
    }

    pub(crate) fn into_hydrated(self) -> FxHashMap<K, Hydrated<V>> {
        self.results
    }

    #[cfg(test)]
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.results.get(key).and_then(Hydrated::value)
    }

    #[cfg(test)]
    pub(crate) fn map<V2>(self, mut f: impl FnMut(V) -> V2) -> HydrationBatch<K, V2> {
        self.map_hydrated(|hydrated| hydrated.map(&mut f))
    }

    #[cfg(test)]
    pub(crate) fn map_hydrated<V2>(
        self,
        mut f: impl FnMut(Hydrated<V>) -> Hydrated<V2>,
    ) -> HydrationBatch<K, V2> {
        HydrationBatch {
            results: self
                .results
                .into_iter()
                .map(|(key, hydrated)| (key, f(hydrated)))
                .collect(),
        }
    }
}

impl<K, V> AsRef<[HydrationBatch<K, V>]> for HydrationBatch<K, V> {
    fn as_ref(&self) -> &[Self] {
        slice::from_ref(self)
    }
}

impl<K: Eq + Hash, V> Default for HydrationBatch<K, V> {
    fn default() -> Self {
        Self {
            results: FxHashMap::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(results: HashMap<u64, Result<Option<u32>, &str>>) -> HydrationBatch<u64, u32> {
        HydrationBatch::from_results(results.keys().copied().collect::<Vec<_>>(), results)
    }

    #[test]
    fn an_expected_key_the_results_leave_out_is_unanswered() {
        let batch: HydrationBatch<u64, u32> =
            HydrationBatch::from_results([1, 2], HashMap::from([(1, Ok::<_, &str>(Some(7)))]));

        assert_eq!(batch.hydrated(&2), Some(&Hydrated::Unanswered));
    }

    #[test]
    fn map_preserves_tri_state() {
        let mapped = batch(HashMap::from([
            (1, Ok(Some(7))),
            (2, Ok(None)),
            (3, Err("boom")),
        ]))
        .map(|v| v * 10);

        assert_eq!(mapped.get(&1), Some(&70));
        assert_eq!(mapped.hydrated(&2), Some(&Hydrated::NotFound));
        assert!(matches!(mapped.hydrated(&3), Some(&Hydrated::Failed)));
    }
}
