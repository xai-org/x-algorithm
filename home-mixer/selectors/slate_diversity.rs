//! Dependency-free slate-diversity selection.
//!
//! Input must already be ranked from highest to lowest score. Keeping this
//! module independent of Home Mixer types lets the standalone local harness
//! compile and test the exact algorithm used by production.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiversityConfig {
    pub selection_size: usize,
    pub author_window_size: usize,
    pub max_posts_per_author: usize,
    pub enable_semantic_diversity: bool,
    pub max_lookahead: usize,
}

pub struct SelectionResult<T> {
    pub selected: Vec<T>,
    pub non_selected: Vec<T>,
}

pub trait SlateItem {
    type AuthorId: Copy + Eq + Hash;

    fn original_author_id(&self) -> Self::AuthorId;
    fn serving_author_id(&self) -> Self::AuthorId;
    fn semantic_ids(&self) -> Option<&[i32]>;
}

pub fn select_from_ranked<T: SlateItem>(
    mut remaining: Vec<T>,
    config: DiversityConfig,
) -> SelectionResult<T> {
    let selection_size = config.selection_size.min(remaining.len());
    let mut selected = Vec::with_capacity(selection_size);
    let mut original_author_counts: HashMap<T::AuthorId, usize> = HashMap::new();
    let mut serving_author_counts: HashMap<T::AuthorId, usize> = HashMap::new();

    while selected.len() < selection_size {
        let previous = selected.last();
        let enforce_author_cap = selected.len() < config.author_window_size;
        let window = &remaining[..remaining.len().min(config.max_lookahead)];
        let strict = window.iter().position(|candidate| {
            (!enforce_author_cap
                || is_within_author_caps(
                    candidate,
                    &original_author_counts,
                    &serving_author_counts,
                    config.max_posts_per_author,
                ))
                && (!config.enable_semantic_diversity
                    || !shares_semantic_cluster(previous, candidate))
        });
        let index = strict
            .or_else(|| {
                window.iter().position(|candidate| {
                    !enforce_author_cap
                        || is_within_author_caps(
                            candidate,
                            &original_author_counts,
                            &serving_author_counts,
                            config.max_posts_per_author,
                        )
                })
            })
            .unwrap_or(0);
        let candidate = remaining.remove(index);
        if enforce_author_cap {
            *original_author_counts
                .entry(candidate.original_author_id())
                .or_default() += 1;
            *serving_author_counts
                .entry(candidate.serving_author_id())
                .or_default() += 1;
        }
        selected.push(candidate);
    }

    SelectionResult {
        selected,
        non_selected: remaining,
    }
}

fn is_within_author_caps<T: SlateItem>(
    candidate: &T,
    original_author_counts: &HashMap<T::AuthorId, usize>,
    serving_author_counts: &HashMap<T::AuthorId, usize>,
    max_posts_per_author: usize,
) -> bool {
    let within_original_cap = original_author_counts
        .get(&candidate.original_author_id())
        .copied()
        .unwrap_or_default()
        < max_posts_per_author;
    let within_serving_cap = serving_author_counts
        .get(&candidate.serving_author_id())
        .copied()
        .unwrap_or_default()
        < max_posts_per_author;
    within_original_cap && within_serving_cap
}

fn shares_semantic_cluster<T: SlateItem>(previous: Option<&T>, candidate: &T) -> bool {
    let Some(previous_ids) = previous.and_then(SlateItem::semantic_ids) else {
        return false;
    };
    let Some(candidate_ids) = candidate.semantic_ids() else {
        return false;
    };

    !previous_ids.is_empty() && previous_ids == candidate_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[derive(Clone, Debug)]
    struct Candidate {
        id: u64,
        original_author_id: u64,
        serving_author_id: u64,
        semantic_ids: Option<Vec<i32>>,
    }

    impl SlateItem for Candidate {
        type AuthorId = u64;

        fn original_author_id(&self) -> Self::AuthorId {
            self.original_author_id
        }

        fn serving_author_id(&self) -> Self::AuthorId {
            self.serving_author_id
        }

        fn semantic_ids(&self) -> Option<&[i32]> {
            self.semantic_ids.as_deref()
        }
    }

    fn candidate(id: u64, author_id: u64, semantic_ids: &[i32]) -> Candidate {
        Candidate {
            id,
            original_author_id: author_id,
            serving_author_id: author_id,
            semantic_ids: Some(semantic_ids.to_vec()),
        }
    }

    fn config(selection_size: usize) -> DiversityConfig {
        DiversityConfig {
            selection_size,
            author_window_size: 20,
            max_posts_per_author: 2,
            enable_semantic_diversity: true,
            max_lookahead: 10,
        }
    }

    fn selected_ids(result: &SelectionResult<Candidate>) -> Vec<u64> {
        result
            .selected
            .iter()
            .map(|candidate| candidate.id)
            .collect()
    }

    #[test]
    fn caps_original_author_in_first_twenty_when_pool_allows() {
        let mut candidates = vec![
            candidate(1, 1, &[1]),
            candidate(2, 1, &[2]),
            candidate(3, 1, &[3]),
        ];
        candidates.extend((2..=20).map(|author| candidate(author + 10, author, &[author as i32])));

        let result = select_from_ranked(candidates, config(22));
        let first_twenty = &result.selected[..20];
        assert_eq!(
            first_twenty
                .iter()
                .filter(|candidate| candidate.original_author_id == 1)
                .count(),
            2
        );
        assert!(!first_twenty.iter().any(|candidate| candidate.id == 3));
    }

    #[test]
    fn caps_serving_author_independently() {
        let mut candidates = vec![
            candidate(1, 1, &[1]),
            candidate(2, 2, &[2]),
            candidate(3, 3, &[3]),
            candidate(4, 4, &[4]),
        ];
        for candidate in &mut candidates[..3] {
            candidate.serving_author_id = 100;
        }

        let result = select_from_ranked(candidates, config(4));
        assert_eq!(selected_ids(&result), vec![1, 2, 4, 3]);
    }

    #[test]
    fn separates_identical_non_empty_semantic_vectors() {
        let result = select_from_ranked(
            vec![
                candidate(1, 1, &[7]),
                candidate(2, 2, &[7]),
                candidate(3, 3, &[8]),
            ],
            config(3),
        );
        assert_eq!(selected_ids(&result), vec![1, 3, 2]);
    }

    #[test]
    fn shared_semantic_component_is_not_the_same_cluster() {
        let result = select_from_ranked(
            vec![
                candidate(1, 1, &[7, 100]),
                candidate(2, 2, &[7, 200]),
                candidate(3, 3, &[8, 300]),
            ],
            config(3),
        );
        assert_eq!(selected_ids(&result), vec![1, 2, 3]);
    }

    #[test]
    fn relaxes_semantics_before_author_caps() {
        let result = select_from_ranked(
            vec![
                candidate(1, 1, &[1]),
                candidate(2, 1, &[2]),
                candidate(3, 2, &[2]),
                candidate(4, 1, &[3]),
            ],
            config(4),
        );
        assert_eq!(selected_ids(&result), vec![1, 2, 3, 4]);
    }

    #[test]
    fn lookahead_bounds_rank_displacement() {
        let mut candidates = vec![candidate(0, 1, &[1])];
        candidates.extend((1..=10).map(|id| candidate(id, id + 1, &[1])));
        candidates.push(candidate(11, 12, &[2]));

        let result = select_from_ranked(candidates, config(12));
        assert_eq!(&selected_ids(&result)[..2], &[0, 1]);
    }

    #[test]
    fn degenerate_configs_preserve_candidates() {
        for config in [
            DiversityConfig {
                selection_size: 3,
                author_window_size: 0,
                max_posts_per_author: 0,
                enable_semantic_diversity: true,
                max_lookahead: 0,
            },
            DiversityConfig {
                selection_size: 3,
                author_window_size: usize::MAX,
                max_posts_per_author: 0,
                enable_semantic_diversity: false,
                max_lookahead: usize::MAX,
            },
        ] {
            let result = select_from_ranked(
                vec![
                    candidate(1, 1, &[1]),
                    candidate(2, 1, &[2]),
                    candidate(3, 1, &[3]),
                ],
                config,
            );
            assert_eq!(selected_ids(&result), vec![1, 2, 3]);
            assert!(result.non_selected.is_empty());
        }
    }

    #[test]
    fn reports_expected_rank_displacement() {
        let result = select_from_ranked(
            vec![
                candidate(1, 1, &[1]),
                candidate(2, 1, &[2]),
                candidate(3, 1, &[3]),
                candidate(4, 2, &[4]),
            ],
            config(4),
        );
        let ids = selected_ids(&result);
        assert_eq!(ids, vec![1, 2, 4, 3]);

        let original_rank: HashMap<u64, usize> = [1, 2, 3, 4]
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        let displacement: Vec<isize> = ids
            .iter()
            .enumerate()
            .map(|(new_rank, id)| new_rank as isize - original_rank[id] as isize)
            .collect();
        assert_eq!(displacement, vec![0, 0, -1, 1]);
    }

    #[test]
    fn randomized_inputs_never_panic_or_lose_candidates() {
        let mut state = 0x4d595df4d0f33173_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };

        for case in 0..500 {
            let candidate_count = (next() % 101) as usize;
            let selection_size = (next() % 121) as usize;
            let candidates: Vec<Candidate> = (0..candidate_count)
                .map(|id| Candidate {
                    id: id as u64,
                    original_author_id: next() % 17,
                    serving_author_id: next() % 17,
                    semantic_ids: (next() % 4 != 0)
                        .then(|| vec![(next() % 6) as i32, (next() % 13) as i32]),
                })
                .collect();
            let expected_ids: HashSet<u64> =
                candidates.iter().map(|candidate| candidate.id).collect();
            let config = DiversityConfig {
                selection_size,
                author_window_size: (next() % 30) as usize,
                max_posts_per_author: (next() % 5) as usize,
                enable_semantic_diversity: next() % 2 == 0,
                max_lookahead: (next() % 20) as usize,
            };

            let result = select_from_ranked(candidates, config);
            let actual_ids: Vec<u64> = result
                .selected
                .iter()
                .chain(&result.non_selected)
                .map(|candidate| candidate.id)
                .collect();
            let actual_set: HashSet<u64> = actual_ids.iter().copied().collect();

            assert_eq!(
                result.selected.len(),
                selection_size.min(candidate_count),
                "case {case}"
            );
            assert_eq!(actual_ids.len(), candidate_count, "case {case}");
            assert_eq!(actual_set.len(), candidate_count, "case {case}");
            assert_eq!(actual_set, expected_ids, "case {case}");
        }
    }
}
