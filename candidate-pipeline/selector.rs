use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
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
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let input_count = candidates.len();
        let result = self.select(query, candidates);
        stats.finish_with_input(input_count);
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
        if std::mem::size_of::<C>() >= 256 && sorted.len() >= 64 {
            let mut order: Vec<_> = sorted
                .iter()
                .enumerate()
                .map(|(index, candidate)| (self.score(candidate), index))
                .collect();
            if !order.iter().any(|(score, _)| score.is_nan()) {
                order.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                let mut destinations = vec![0; sorted.len()];
                for (target, (_, source)) in order.into_iter().enumerate() {
                    destinations[source] = target;
                }
                for index in 0..sorted.len() {
                    while destinations[index] != index {
                        let next = destinations[index];
                        sorted.swap(index, next);
                        destinations.swap(index, next);
                    }
                }
                return sorted;
            }
        }
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use xai_feature_switches::Params;

    #[derive(Clone, Default)]
    struct Query(Params);

    impl PipelineQuery for Query {
        fn params(&self) -> &Params {
            &self.0
        }
        fn decider(&self) -> Option<&xai_decider::Decider> {
            None
        }
    }

    #[derive(Clone)]
    struct Candidate {
        id: usize,
        score: f64,
        payload: [u8; 1024],
    }

    #[derive(Default)]
    struct CountedSelector(AtomicUsize);

    impl Selector<Query, Candidate> for CountedSelector {
        fn score(&self, candidate: &Candidate) -> f64 {
            self.0.fetch_add(1, Ordering::Relaxed);
            candidate.score
        }
    }

    #[test]
    fn large_candidate_sort_extracts_each_score_once_and_preserves_payloads() {
        let input: Vec<_> = (0..3000)
            .map(|id| Candidate {
                id,
                score: ((id * 691) % 3000) as f64,
                payload: [id as u8; 1024],
            })
            .collect();
        let count = input.len();
        let selector = CountedSelector::default();
        let result = selector.run(&Query::default(), input, PipelineStage::Selector);
        assert_eq!(selector.0.load(Ordering::Relaxed), count);
        assert_eq!(result.selected.len(), count);
        assert!(result.non_selected.is_empty());
        assert!(result
            .selected
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score));
        let ids: std::collections::HashSet<_> = result.selected.iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), count);
        for candidate in result.selected {
            assert!(candidate.id < count);
            assert_eq!(candidate.score, ((candidate.id * 691) % count) as f64);
            assert_eq!(candidate.payload, [candidate.id as u8; 1024]);
        }
    }

    #[test]
    fn nan_scores_keep_the_legacy_comparator_path_and_candidate_bits() {
        let mut input: Vec<_> = (0..3000)
            .map(|id| Candidate {
                id,
                score: (3000 - id) as f64,
                payload: [id as u8; 1024],
            })
            .collect();
        input[1500].score = f64::NAN;
        let expected: Vec<_> = input.iter().map(|c| (c.id, c.score.to_bits())).collect();
        let selector = CountedSelector::default();
        let result = selector.run(&Query::default(), input, PipelineStage::Selector);
        assert_eq!(
            result
                .selected
                .iter()
                .map(|c| (c.id, c.score.to_bits()))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(selector.0.load(Ordering::Relaxed) > expected.len());
        for candidate in result.selected {
            assert_eq!(candidate.payload, [candidate.id as u8; 1024]);
        }
    }
}
