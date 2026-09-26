use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::util;
use crate::SPAN_LEVEL;
use std::any::type_name_of_val;
use tracing::{field::Empty, Span};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

pub struct SelectResult<C> {
    pub selected: Vec<C>,
    pub non_selected: Vec<C>,
}

impl<C> SelectResult<C> {
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty() && self.non_selected.is_empty()
    }
}

pub trait Selector<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "selector", fields(
        name = self.name(),
        input_count = candidates.len(),
        selected_count = Empty,
        non_selected_count = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>, stage: PipelineStage) -> SelectResult<C> {
        let result = self.select(query, candidates);
        let span = Span::current();
        span.record("selected_count", result.selected.len());
        span.record("non_selected_count", result.non_selected.len());
        self.stat(&result, stage);
        result
    }

    fn select(&self, _query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            let non_selected = sorted.split_off(limit.min(sorted.len()));
            SelectResult {
                selected: sorted,
                non_selected,
            }
        } else {
            SelectResult {
                selected: sorted,
                non_selected: vec![],
            }
        }
    }

    fn score(&self, candidate: &C) -> f64;

    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| cmp_ranking_scores_desc(self.score(a), self.score(b)));
        sorted
    }

    fn size(&self) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat(&self, result: &SelectResult<C>, stage: PipelineStage) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.run", self.name());
            let result_size = result.len() as f64;
            receiver.observe(
                metric_name.as_str(),
                &stage.stat_labels(self.name(), "result_size"),
                result_size,
                HistogramBuckets::Bucket0To50,
            );
            if result_size == 0.0 {
                receiver.incr(
                    metric_name.as_str(),
                    &stage.stat_labels(self.name(), "result_empty"),
                    1u64,
                );
            }
        }
    }
}

/// Finite scores keep their value. Non-finite scores (`NaN`, `±inf`) sink to
/// the bottom of a descending ranking so they cannot occupy a Top-K slot.
fn normalized_ranking_score(score: f64) -> f64 {
    if score.is_finite() {
        score
    } else {
        f64::NEG_INFINITY
    }
}

fn cmp_ranking_scores_desc(left: f64, right: f64) -> std::cmp::Ordering {
    normalized_ranking_score(right).total_cmp(&normalized_ranking_score(left))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rank(mut scores: Vec<f64>) -> Vec<f64> {
        scores.sort_by(|&a, &b| cmp_ranking_scores_desc(a, b));
        scores
    }

    fn top_k(scores: Vec<f64>, k: usize) -> Vec<f64> {
        let ranked = rank(scores);
        ranked.into_iter().take(k).collect()
    }

    fn assert_finite_prefix(actual: &[f64], expected: &[f64]) {
        assert!(actual.len() >= expected.len());
        assert_eq!(&actual[..expected.len()], expected);
        assert!(actual[expected.len()..].iter().all(|s| !s.is_finite()));
    }

    #[test]
    fn nan_ranks_below_finite_scores() {
        let ranked = rank(vec![0.90, f64::NAN, 0.80]);
        assert_eq!(ranked[0], 0.90);
        assert_eq!(ranked[1], 0.80);
        assert!(ranked[2].is_nan());
    }

    #[test]
    fn top_k_does_not_keep_nan_over_finite_candidate() {
        let selected = top_k(vec![0.90, f64::NAN, 0.80], 2);
        assert_eq!(selected, vec![0.90, 0.80]);
    }

    #[test]
    fn non_finite_scores_rank_below_finite_scores() {
        let ranked = rank(vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.90,
            0.80,
        ]);
        assert_finite_prefix(&ranked, &[0.90, 0.80]);
    }

    #[test]
    fn finite_scores_keep_descending_order() {
        let ranked = rank(vec![0.5, -1.0, 2.0, 0.0, -0.25]);
        assert_eq!(ranked, vec![2.0, 0.5, 0.0, -0.25, -1.0]);
    }
}
