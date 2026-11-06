use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use crate::SPAN_LEVEL;
use std::any::{type_name_of_val, Any};
use tracing::{field::Empty, Span};
use xai_stats_receiver::global_stats_receiver;

pub struct FilterResult<C> {
    pub kept: Vec<C>,
    pub removed: Vec<C>,
}

impl<C> FilterResult<C> {
    pub fn partition(mut candidates: Vec<C>, mut keep: impl FnMut(&C) -> bool) -> Self {
        let removed = candidates
            .extract_if(.., |candidate| !keep(candidate))
            .collect();
        if candidates.is_empty() {
            candidates = Vec::new();
        }
        Self {
            kept: candidates,
            removed,
        }
    }
}

pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "filter", fields(
        name = self.name(),
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>, stage: PipelineStage) -> FilterResult<C> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let result = self.filter(query, candidates);
        stats.finish_filter(result.kept.len(), result.removed.len());
        let total = result.kept.len() + result.removed.len();
        let rate = if total > 0 {
            result.removed.len() as f64 / total as f64
        } else {
            0.0
        };
        let span = Span::current();
        span.record("kept_count", result.kept.len());
        span.record("removed_count", result.removed.len());
        span.record("filter_rate", format!("{:.3}", rate).as_str());
        self.stat(&result, stage);
        result
    }

    fn filter(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat(&self, result: &FilterResult<C>, stage: PipelineStage) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.run", self.name());
            receiver.incr(
                metric_name.as_str(),
                &stage.stat_labels(self.name(), "kept"),
                result.kept.len() as u64,
            );
            receiver.incr(
                metric_name.as_str(),
                &stage.stat_labels(self.name(), "removed"),
                result.removed.len() as u64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_preserves_input_order_and_calls_the_predicate_once_per_item() {
        let input: Vec<_> = (0..17).map(|id| (id, format!("payload-{id}"))).collect();
        let expected_kept: Vec<_> = input
            .iter()
            .filter(|(id, _)| id % 2 == 0)
            .cloned()
            .collect();
        let expected_removed: Vec<_> = input
            .iter()
            .filter(|(id, _)| id % 2 != 0)
            .cloned()
            .collect();
        let mut visited = Vec::new();
        let result = FilterResult::partition(input, |(id, _)| {
            visited.push(*id);
            id % 2 == 0
        });
        assert_eq!(visited, (0..17).collect::<Vec<_>>());
        assert_eq!(result.kept, expected_kept);
        assert_eq!(result.removed, expected_removed);
    }

    #[test]
    fn partition_accepts_nonclone_candidates_and_releases_empty_kept_buffers() {
        struct NonClone(String);
        let input = vec![
            NonClone("kept".to_string()),
            NonClone("removed".to_string()),
        ];
        let result = FilterResult::partition(input, |candidate| candidate.0 == "kept");
        assert_eq!(result.kept[0].0, "kept");
        assert_eq!(result.removed[0].0, "removed");
        let result = FilterResult::partition(result.kept, |_| false);
        assert!(result.kept.is_empty());
        assert_eq!(result.kept.capacity(), 0);
        assert_eq!(result.removed[0].0, "kept");
    }
}
