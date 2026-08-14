use super::slate_diversity::{self, DiversityConfig, SlateItem};
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params;
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
/// Author caps apply near the top; semantic adjacency applies throughout.
/// Lookahead bounds how far a constraint may displace a ranked candidate.
pub struct SlateDiversitySelector;

impl SlateItem for PostCandidate {
    type AuthorId = u64;

    fn original_author_id(&self) -> Self::AuthorId {
        self.get_original_author_id()
    }

    fn serving_author_id(&self) -> Self::AuthorId {
        self.author_id
    }

    fn semantic_ids(&self) -> Option<&[i32]> {
        self.semantic_ids.as_deref()
    }
}

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

        let result = slate_diversity::select_from_ranked(
            self.sort(candidates),
            DiversityConfig {
                selection_size: params::TOP_K_CANDIDATES_TO_SELECT,
                author_window_size: query.params.get(params::SlateDiversityAuthorWindowSize)
                    as usize,
                max_posts_per_author: query.params.get(params::SlateDiversityMaxPostsPerAuthor)
                    as usize,
                enable_semantic_diversity: query.params.get(params::EnableSlateSemanticDiversity),
                max_lookahead: query.params.get(params::SlateDiversityMaxLookahead) as usize,
            },
        );

        SelectResult {
            selected: result.selected,
            non_selected: result.non_selected,
        }
    }
}
