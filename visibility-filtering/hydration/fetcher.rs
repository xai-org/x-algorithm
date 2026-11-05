use crate::hydration::batch::{Hydrated, HydrationBatch};
use crate::hydration::inventory::Answer;
use crate::hydration::relation::Relation;
use crate::hydration::sources::Source;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;
use strum::VariantArray;

pub(super) struct Fetcher<V, K = u64> {
    states: FxHashMap<K, State<V>>,
}

enum State<V> {
    Pending,
    Landed(Hydrated<V>),
}

impl<V> State<V> {
    fn is_complete(&self) -> bool {
        match self {
            State::Landed(answer) => answer.is_complete(),
            State::Pending => false,
        }
    }
}

impl<V, K> Default for Fetcher<V, K> {
    fn default() -> Self {
        Self {
            states: FxHashMap::default(),
        }
    }
}

impl<V, K: Copy + Eq + Hash> Fetcher<V, K> {
    pub(super) fn land(&mut self, claimed: &[K], batch: HydrationBatch<K, V>) {
        let mut answers = batch.into_hydrated();
        for &key in claimed {
            let answer = answers.remove(&key).unwrap_or(Hydrated::Unanswered);
            self.states.insert(key, State::Landed(answer));
        }
    }

    pub(super) fn get(&self, key: K) -> Option<&V> {
        match self.states.get(&key)? {
            State::Landed(answer) => answer.value(),
            State::Pending => None,
        }
    }

    pub(super) fn take(&mut self, key: K) -> Option<V> {
        match self.states.remove(&key)? {
            State::Landed(answer) => answer.into_value(),
            State::Pending => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Owner {
    Request,
    Related(Relation),
}

#[derive(Clone, Copy)]
pub(super) enum AskedFields {
    Planned,
    EveryLevel,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FetcherId {
    pub(crate) source: Source,
    pub(crate) owner: Owner,
}

impl FetcherId {
    #[cfg(test)]
    pub(crate) fn all() -> impl Iterator<Item = Self> {
        Source::VARIANTS.iter().flat_map(|&source| {
            let owners = Relation::VARIANTS
                .iter()
                .filter(move |relation| relation.owns(source))
                .map(|&relation| Owner::Related(relation));
            std::iter::once(Owner::Request)
                .chain(owners)
                .map(move |owner| Self { source, owner })
        })
    }

    pub(super) const fn asked_fields(self) -> AskedFields {
        match (self.owner, self.source) {
            (Owner::Request, Source::GizmoduckAuthor | Source::GizmoduckActor) => {
                AskedFields::EveryLevel
            }
            _ => AskedFields::Planned,
        }
    }
}

#[cfg(test)]
impl From<Source> for FetcherId {
    fn from(source: Source) -> Self {
        Self {
            source,
            owner: Owner::Request,
        }
    }
}

#[cfg(test)]
impl From<(Relation, Source)> for FetcherId {
    fn from((relation, source): (Relation, Source)) -> Self {
        Self {
            source,
            owner: Owner::Related(relation),
        }
    }
}

impl fmt::Debug for FetcherId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.owner {
            Owner::Request => write!(f, "{:?}", self.source),
            Owner::Related(relation) => write!(f, "{:?}@{}", self.source, <&str>::from(relation)),
        }
    }
}

pub(super) struct Fetchers<V, K = u64> {
    request: Fetcher<V, K>,
    related: [Fetcher<V, K>; Relation::VARIANTS.len()],
}

impl<V, K> Default for Fetchers<V, K> {
    fn default() -> Self {
        Self {
            request: Fetcher::default(),
            related: Default::default(),
        }
    }
}

impl<V, K> Fetchers<V, K> {
    pub(super) fn request(&self) -> &Fetcher<V, K> {
        &self.request
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "Relation::VARIANTS sizes the array, so every Relation indexes it"
    )]
    pub(super) fn of(&self, owner: Owner) -> &Fetcher<V, K> {
        match owner {
            Owner::Request => &self.request,
            Owner::Related(relation) => &self.related[relation as usize],
        }
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "Relation::VARIANTS sizes the array, so every Relation indexes it"
    )]
    pub(super) fn of_mut(&mut self, owner: Owner) -> &mut Fetcher<V, K> {
        match owner {
            Owner::Request => &mut self.request,
            Owner::Related(relation) => &mut self.related[relation as usize],
        }
    }
}

pub(super) trait AnyFetcher<K = u64> {
    fn claim(&mut self, keys: Vec<K>) -> Vec<K>;
    fn has_claimed(&self) -> bool;
    fn is_claimed(&self, key: K) -> bool;
    fn is_incomplete(&self, key: K) -> bool;
    fn answer(&self, key: K) -> Answer;
}

impl<V, K: Copy + Eq + Hash> AnyFetcher<K> for Fetcher<V, K> {
    fn claim(&mut self, mut keys: Vec<K>) -> Vec<K> {
        self.states.reserve(keys.len());
        keys.retain(|&key| match self.states.entry(key) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(State::Pending);
                true
            }
        });
        keys
    }

    fn has_claimed(&self) -> bool {
        !self.states.is_empty()
    }

    fn is_claimed(&self, key: K) -> bool {
        self.states.contains_key(&key)
    }

    fn is_incomplete(&self, key: K) -> bool {
        self.states
            .get(&key)
            .is_some_and(|state| !state.is_complete())
    }

    fn answer(&self, key: K) -> Answer {
        match self.states.get(&key) {
            Some(State::Landed(Hydrated::Found(_))) => Answer::Found,
            Some(State::Landed(Hydrated::Partial(_))) => Answer::Partial,
            Some(State::Landed(Hydrated::NotFound)) => Answer::NotFound,
            Some(State::Landed(Hydrated::Failed)) => Answer::Failed,
            Some(State::Landed(Hydrated::Unanswered) | State::Pending) => Answer::Unanswered,
            None => Answer::Unclaimed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_found_and_not_found_keys_are_complete() {
        let mut fetcher = Fetcher::default();
        let keys = fetcher.claim(vec![1, 2, 3, 4, 5]);
        fetcher.land(
            &keys[..4],
            HydrationBatch::from_hydrated(FxHashMap::from_iter([
                (1, Hydrated::Found(7)),
                (2, Hydrated::NotFound),
                (3, Hydrated::Partial(7)),
                (4, Hydrated::Failed),
            ])),
        );

        let incomplete: Vec<u64> = keys
            .into_iter()
            .filter(|&key| fetcher.is_incomplete(key))
            .collect();
        assert_eq!(incomplete, [3, 4, 5]);
        assert_eq!(fetcher.get(3), Some(&7));
    }

    #[test]
    fn a_claimed_key_the_answer_leaves_out_lands_unanswered() {
        let mut fetcher = Fetcher::default();
        let keys = fetcher.claim(vec![1, 2]);
        fetcher.land(
            &keys,
            HydrationBatch::from_hydrated(FxHashMap::from_iter([(1, Hydrated::Found(7))])),
        );

        assert!(fetcher.is_incomplete(2));
        assert_eq!(fetcher.answer(2), Answer::Unanswered);
    }
}
