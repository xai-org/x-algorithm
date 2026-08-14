use crate::models::candidate::CandidateHelpers;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use rustc_hash::FxHashMap;
use xai_candidate_pipeline::selector::{SelectResult, Selector};

pub struct TopKScoreSelector;

impl Selector<ScoredPostsQuery, PostCandidate> for TopKScoreSelector {
    fn score(&self, candidate: &PostCandidate) -> f64 {
        candidate.score.unwrap_or(f64::NEG_INFINITY)
    }
    fn size(&self) -> Option<usize> {
        Some(params::TOP_K_CANDIDATES_TO_SELECT)
    }
}

/// Selects the highest-scoring candidates while improving slate diversity.
/// The author cap applies near the top; semantic adjacency applies throughout.
///
/// Constraints are applied greedily in score order. When the remaining pool
/// cannot satisfy every constraint, semantic adjacency is relaxed first and
/// the author cap second. This keeps the response full while making any
/// constraint violation an explicit best-effort fallback.
pub struct SlateDiversitySelector;

impl Selector<ScoredPostsQuery, PostCandidate> for SlateDiversitySelector {
    fn score(&self, candidate: &PostCandidate) -> f64 {
        candidate.score.unwrap_or(f64::NEG_INFINITY)
    }

    fn select(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> SelectResult<PostCandidate> {
        if !query.params.get(params::EnableSlateDiversity) {
            return TopKScoreSelector.select(query, candidates);
        }

        let mut remaining = self.sort(candidates);
        let selection_size = params::TOP_K_CANDIDATES_TO_SELECT.min(remaining.len());
        let author_window = query.params.get(params::SlateDiversityAuthorWindowSize) as usize;
        let max_posts_per_author =
            query.params.get(params::SlateDiversityMaxPostsPerAuthor) as usize;
        let enable_semantic_diversity = query.params.get(params::EnableSlateSemanticDiversity);
        let mut selected = Vec::with_capacity(selection_size);
        let mut author_counts: FxHashMap<u64, usize> = FxHashMap::default();

        while selected.len() < selection_size {
            let previous = selected.last();
            let enforce_author_cap = selected.len() < author_window;
            let strict = remaining.iter().position(|candidate| {
                (!enforce_author_cap
                    || is_within_author_cap(candidate, &author_counts, max_posts_per_author))
                    && (!enable_semantic_diversity || !shares_semantic_cluster(previous, candidate))
            });
            let author_only = remaining.iter().position(|candidate| {
                !enforce_author_cap
                    || is_within_author_cap(candidate, &author_counts, max_posts_per_author)
            });
            let index = strict.or(author_only).unwrap_or(0);
            let candidate = remaining.remove(index);
            if enforce_author_cap {
                *author_counts
                    .entry(candidate.get_original_author_id())
                    .or_default() += 1;
            }
            selected.push(candidate);
        }

        SelectResult {
            selected,
            non_selected: remaining,
        }
    }
}

fn is_within_author_cap(
    candidate: &PostCandidate,
    author_counts: &FxHashMap<u64, usize>,
    max_posts_per_author: usize,
) -> bool {
    author_counts
        .get(&candidate.get_original_author_id())
        .copied()
        .unwrap_or_default()
        < max_posts_per_author
}

fn shares_semantic_cluster(previous: Option<&PostCandidate>, candidate: &PostCandidate) -> bool {
    let Some(previous_ids) = previous.and_then(|post| post.semantic_ids.as_ref()) else {
        return false;
    };
    let Some(candidate_ids) = candidate.semantic_ids.as_ref() else {
        return false;
    };

    previous_ids
        .iter()
        .any(|semantic_id| candidate_ids.contains(semantic_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(tweet_id: u64, author_id: u64, score: f64, semantic_ids: &[i32]) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            score: Some(score),
            semantic_ids: Some(semantic_ids.to_vec()),
            ..Default::default()
        }
    }

    fn select(candidates: Vec<PostCandidate>) -> SelectResult<PostCandidate> {
        SlateDiversitySelector.select(&ScoredPostsQuery::default(), candidates)
    }

    #[test]
    fn caps_each_author_at_two_in_first_twenty_when_pool_allows() {
        let mut candidates = vec![
            candidate(1, 1, 100.0, &[1]),
            candidate(2, 1, 99.0, &[2]),
            candidate(3, 1, 98.0, &[3]),
        ];
        candidates.extend(
            (2..=11).map(|author| {
                candidate(author + 10, author, 90.0 - author as f64, &[author as i32])
            }),
        );
        candidates.extend(
            (12..=20).map(|author| {
                candidate(author + 10, author, 50.0 - author as f64, &[author as i32])
            }),
        );

        let result = select(candidates);
        let first_twenty = &result.selected[..20];
        let author_one_count = first_twenty
            .iter()
            .filter(|candidate| candidate.author_id == 1)
            .count();

        assert_eq!(author_one_count, 2);
        assert!(!first_twenty.iter().any(|candidate| candidate.tweet_id == 3));
    }

    #[test]
    fn avoids_adjacent_shared_semantic_clusters() {
        let result = select(vec![
            candidate(1, 1, 10.0, &[7]),
            candidate(2, 2, 9.0, &[7]),
            candidate(3, 3, 8.0, &[8]),
        ]);

        let ids: Vec<u64> = result
            .selected
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect();
        assert_eq!(ids, vec![1, 3, 2]);
    }

    #[test]
    fn relaxes_semantic_constraint_before_author_cap() {
        let result = select(vec![
            candidate(1, 1, 10.0, &[1]),
            candidate(2, 1, 9.0, &[2]),
            candidate(3, 2, 8.0, &[2]),
            candidate(4, 1, 7.0, &[3]),
        ]);

        let ids: Vec<u64> = result
            .selected
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn falls_back_to_score_order_when_author_cap_is_impossible() {
        let result = select(vec![
            candidate(1, 1, 10.0, &[1]),
            candidate(2, 1, 9.0, &[2]),
            candidate(3, 1, 8.0, &[3]),
        ]);

        let ids: Vec<u64> = result
            .selected
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn retains_top_k_size_and_reports_non_selected_candidates() {
        let candidates = (0..60)
            .map(|index| candidate(index, index, 100.0 - index as f64, &[index as i32]))
            .collect();

        let result = select(candidates);

        assert_eq!(result.selected.len(), params::TOP_K_CANDIDATES_TO_SELECT);
        assert_eq!(result.non_selected.len(), 10);
        assert_eq!(result.selected[0].score, Some(100.0));
    }
}
