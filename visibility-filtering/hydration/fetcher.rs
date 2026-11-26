use crate::hydration::batch::{Hydrated, HydrationBatch};
use crate::hydration::inventory::Answer;
use crate::hydration::relation::Relation;
use crate::hydration::sources::Source;
#[cfg(test)]
use crate::hydration::Hydrator;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::Hash;
use strum::VariantArray;

pub(super) struct Fetcher<V, K = u64> {
    states: FxHashMap<K, Hydrated<V>>,
}

impl<V, K> Default for Fetcher<V, K> {
    fn default() -> Self {
        Self {
            states: FxHashMap::default(),
        }
    }
}

impl<V, K: Copy + Eq + Hash> Fetcher<V, K> {
    pub(super) fn land(&mut self, batch: HydrationBatch<K, V>) {
        for (key, answer) in batch.into_hydrated() {
            if let Some(state) = self.states.get_mut(&key) {
                debug_assert!(
                    matches!(state, Hydrated::Unanswered),
                    "a batch answers only its call's keys"
                );
                *state = answer;
            }
        }
    }

    pub(super) fn land_by<S: Copy + Eq + Hash>(
        &mut self,
        claimed: &[S],
        batch: HydrationBatch<S, V>,
        key: impl Fn(S) -> K,
    ) {
        let mut answers = batch.into_hydrated();
        for &claimed in claimed {
            let answer = answers.remove(&claimed).unwrap_or(Hydrated::Unanswered);
            self.states.insert(key(claimed), answer);
        }
    }

    pub(super) fn claim_by<S>(&mut self, mut keys: Vec<S>, key: impl Fn(&S) -> K) -> Vec<S> {
        self.states.reserve(keys.len());
        keys.retain(|claimed| match self.states.entry(key(claimed)) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(Hydrated::Unanswered);
                true
            }
        });
        keys
    }

    pub(super) fn get(&self, key: K) -> Option<&V> {
        self.states.get(&key)?.value()
    }

    pub(super) fn take(&mut self, key: K) -> Option<V> {
        self.states.remove(&key)?.into_value()
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
        let related = Hydrator::VARIANTS
            .iter()
            .map(|node| node.spec().fetcher())
            .filter(|fetcher| fetcher.owner != Owner::Request);
        let mut all: Vec<Self> = Vec::new();
        for fetcher in Source::VARIANTS
            .iter()
            .map(|&source| Self::from(source))
            .chain(related)
        {
            if !all.contains(&fetcher) {
                all.push(fetcher);
            }
        }
        all.into_iter()
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
    fn is_incomplete(&self, key: K) -> bool;
    fn answer(&self, key: K) -> Answer;
}

impl<V, K: Copy + Eq + Hash> AnyFetcher<K> for Fetcher<V, K> {
    fn claim(&mut self, keys: Vec<K>) -> Vec<K> {
        self.claim_by(keys, |&key| key)
    }

    fn has_claimed(&self) -> bool {
        !self.states.is_empty()
    }

    fn is_incomplete(&self, key: K) -> bool {
        self.states
            .get(&key)
            .is_some_and(|answer| !answer.is_complete())
    }

    fn answer(&self, key: K) -> Answer {
        match self.states.get(&key) {
            Some(Hydrated::Found(_)) => Answer::Found,
            Some(Hydrated::Partial(_)) => Answer::Partial,
            Some(Hydrated::NotFound) => Answer::NotFound,
            Some(Hydrated::Failed) => Answer::Failed,
            Some(Hydrated::Unanswered) => Answer::Unanswered,
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
        fetcher.land(HydrationBatch::from_hydrated(FxHashMap::from_iter([
            (1, Hydrated::Found(7)),
            (2, Hydrated::NotFound),
            (3, Hydrated::Partial(7)),
            (4, Hydrated::Failed),
        ])));

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
        assert_eq!(fetcher.claim(vec![1, 2]), [1, 2]);
        fetcher.land(HydrationBatch::from_hydrated(FxHashMap::from_iter([(
            1,
            Hydrated::Found(7),
        )])));

        assert!(fetcher.is_incomplete(2));
        assert_eq!(fetcher.answer(2), Answer::Unanswered);
    }
}
