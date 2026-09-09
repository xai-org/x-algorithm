use crate::models::{AuthorId, TweetId};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HydrationError {
    MissingResponse,
    Timeout,
    Rpc(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Hydrated<V> {
    Found(V),
    NotFound,
    Failed(HydrationError),
}

impl<V> Hydrated<V> {
    pub(crate) fn value(&self) -> Option<&V> {
        match self {
            Hydrated::Found(value) => Some(value),
            Hydrated::NotFound | Hydrated::Failed(_) => None,
        }
    }
}

impl<V, E: Display> From<Result<Option<V>, E>> for Hydrated<V> {
    fn from(result: Result<Option<V>, E>) -> Self {
        match result {
            Ok(Some(value)) => Hydrated::Found(value),
            Ok(None) => Hydrated::NotFound,
            Err(e) => Hydrated::Failed(HydrationError::Rpc(e.to_string())),
        }
    }
}

pub(crate) struct HydrationBatch<K, V> {
    results: HashMap<K, Hydrated<V>>,
}

pub(crate) type TweetHydrationBatch<V> = HydrationBatch<TweetId, V>;
pub(crate) type AuthorHydrationBatch<V> = HydrationBatch<AuthorId, V>;

impl<K: Eq + Hash, V> HydrationBatch<K, V> {
    pub(crate) fn empty() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    pub(crate) fn from_results<E: Display>(
        expected: impl IntoIterator<Item = K>,
        mut results: HashMap<K, Result<Option<V>, E>>,
    ) -> Self {
        Self::from_expected(expected, |key| {
            results
                .remove(key)
                .map(Hydrated::from)
                .unwrap_or(Hydrated::Failed(HydrationError::MissingResponse))
        })
    }

    pub(crate) fn from_values(
        expected: impl IntoIterator<Item = K>,
        mut values: HashMap<K, V>,
    ) -> Self {
        Self::from_expected(expected, |key| {
            values
                .remove(key)
                .map(Hydrated::Found)
                .unwrap_or(Hydrated::Failed(HydrationError::MissingResponse))
        })
    }

    pub(crate) fn from_hydrated(results: HashMap<K, Hydrated<V>>) -> Self {
        Self { results }
    }

    fn from_expected(
        expected: impl IntoIterator<Item = K>,
        mut resolve: impl FnMut(&K) -> Hydrated<V>,
    ) -> Self {
        let mut results = HashMap::new();
        for key in expected {
            if let Entry::Vacant(entry) = results.entry(key) {
                let hydrated = resolve(entry.key());
                entry.insert(hydrated);
            }
        }
        Self { results }
    }

    pub(crate) fn timed_out(expected: impl IntoIterator<Item = K>) -> Self {
        Self {
            results: expected
                .into_iter()
                .map(|key| (key, Hydrated::Failed(HydrationError::Timeout)))
                .collect(),
        }
    }

    pub(crate) fn is_failed(&self, key: &K) -> bool {
        self.results.get(key).is_some_and(Hydrated::is_failed)
    }

    pub(crate) fn hydrated(&self, key: &K) -> Option<&Hydrated<V>> {
        self.results.get(key)
    }

    pub(crate) fn into_hydrated(self) -> HashMap<K, Hydrated<V>> {
        self.results
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.results.get(key).and_then(Hydrated::value)
    }

    pub(crate) fn get_or_default(&self, key: &K) -> V
    where
        V: Clone + Default,
    {
        self.get(key).cloned().unwrap_or_default()
    }

    pub(crate) fn map_keys<K2: Eq + Hash>(self, f: impl Fn(K) -> K2) -> HydrationBatch<K2, V> {
        HydrationBatch {
            results: self
                .results
                .into_iter()
                .map(|(key, hydrated)| (f(key), hydrated))
                .collect(),
        }
    }

    pub(crate) fn map<V2>(self, f: impl Fn(V) -> V2) -> HydrationBatch<K, V2> {
        HydrationBatch {
            results: self
                .results
                .into_iter()
                .map(|(key, hydrated)| {
                    let hydrated = match hydrated {
                        Hydrated::Found(value) => Hydrated::Found(f(value)),
                        Hydrated::NotFound => Hydrated::NotFound,
                        Hydrated::Failed(e) => Hydrated::Failed(e),
                    };
                    (key, hydrated)
                })
                .collect(),
        }
    }

    pub(crate) fn project<K2: Eq + Hash>(
        &self,
        pairs: impl IntoIterator<Item = (K2, K)>,
    ) -> HydrationBatch<K2, V>
    where
        V: Clone,
    {
        HydrationBatch {
            results: pairs
                .into_iter()
                .map(|(new_key, key)| {
                    let hydrated = match self.results.get(&key) {
                        Some(hydrated) => hydrated.clone(),
                        None => Hydrated::Failed(HydrationError::MissingResponse),
                    };
                    (new_key, hydrated)
                })
                .collect(),
        }
    }
}

impl<K: Eq + Hash, V> Default for HydrationBatch<K, V> {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(results: HashMap<u64, Result<Option<u32>, &str>>) -> HydrationBatch<u64, u32> {
        HydrationBatch::from_results(results.keys().copied().collect::<Vec<_>>(), results)
    }

    #[test]
    fn found_not_found_and_failed_stay_distinguishable() {
        let batch = batch(HashMap::from([
            (1, Ok(Some(7))),
            (2, Ok(None)),
            (3, Err("backend unavailable")),
        ]));

        assert_eq!(batch.get(&1), Some(&7));
        assert_eq!(batch.hydrated(&2), Some(&Hydrated::NotFound));
        assert_eq!(
            batch.hydrated(&3),
            Some(&Hydrated::Failed(HydrationError::Rpc(
                "backend unavailable".into()
            )))
        );
        assert_eq!(batch.get_or_default(&2), 0);
        assert_eq!(batch.get_or_default(&3), 0);
    }

    #[test]
    fn absent_expected_keys_become_missing() {
        let batch: HydrationBatch<u64, u32> =
            HydrationBatch::from_results([1, 2], HashMap::from([(1, Ok::<_, &str>(Some(7)))]));

        assert_eq!(
            batch.hydrated(&2),
            Some(&Hydrated::Failed(HydrationError::MissingResponse))
        );
    }

    #[test]
    fn timeout_fails_every_expected_key() {
        let batch: HydrationBatch<u64, u32> = HydrationBatch::timed_out([1, 2]);

        assert_eq!(
            batch.hydrated(&1),
            Some(&Hydrated::Failed(HydrationError::Timeout))
        );
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
        assert!(matches!(mapped.hydrated(&3), Some(&Hydrated::Failed(_))));
    }

    #[test]
    fn project_fans_source_results_out_to_new_keys() {
        let by_author = batch(HashMap::from([(10, Ok(Some(7))), (20, Err("boom"))]));

        let by_tweet = by_author.project([(TweetId(1), 10), (TweetId(2), 10), (TweetId(3), 20)]);

        assert_eq!(by_tweet.get(&TweetId(1)), Some(&7));
        assert_eq!(by_tweet.get(&TweetId(2)), Some(&7));
        assert!(matches!(
            by_tweet.hydrated(&TweetId(3)),
            Some(&Hydrated::Failed(_))
        ));
    }

    #[test]
    fn project_unknown_source_key_is_missing() {
        let by_author: HydrationBatch<u64, u32> = HydrationBatch::empty();

        let by_tweet = by_author.project([(TweetId(1), 10)]);

        assert_eq!(
            by_tweet.hydrated(&TweetId(1)),
            Some(&Hydrated::Failed(HydrationError::MissingResponse))
        );
    }

    #[test]
    fn duplicate_expected_keys_resolve_once() {
        let batch: HydrationBatch<u64, u32> =
            HydrationBatch::from_values([1, 1, 2, 2], HashMap::from([(1, 7), (2, 8)]));

        assert_eq!(batch.get(&1), Some(&7));
        assert_eq!(batch.get(&2), Some(&8));
    }

    #[test]
    fn from_values_marks_present_found_and_absent_missing() {
        let batch: HydrationBatch<u64, u32> =
            HydrationBatch::from_values([1, 2], HashMap::from([(1, 7)]));

        assert_eq!(batch.get(&1), Some(&7));
        assert_eq!(
            batch.hydrated(&2),
            Some(&Hydrated::Failed(HydrationError::MissingResponse))
        );
    }
}
