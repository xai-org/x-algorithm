use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use crate::SPAN_LEVEL;
use std::any::type_name_of_val;
use tonic::async_trait;
use tracing::warn;

#[async_trait]
pub trait Scorer<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency = Vm)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "scorer", fields(name = self.name()))]
    async fn run(
        &self,
        query: &Q,
        candidates: &[C],
        stage: PipelineStage,
    ) -> Vec<Result<C, String>> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let scored = self.score(query, candidates).await;
        let expected_len = candidates.len();
        let results = if scored.len() == expected_len {
            scored
        } else {
            let message = format!(
                "Scorer length_mismatch expected={} got={}",
                expected_len,
                scored.len()
            );
            warn!(
                "Skipped: length_mismatch expected={} got={}",
                expected_len,
                scored.len()
            );
            vec![Err(message); expected_len]
        };
        stats.finish_with_results(&results);
        results
    }

    async fn score(&self, query: &Q, candidates: &[C]) -> Vec<Result<C, String>>;

    fn update(&self, candidate: &mut C, scored: C);

    fn update_all(&self, candidates: &mut [C], scored: Vec<Result<C, String>>) {
        for (candidate, scored) in candidates.iter_mut().zip(scored) {
            if let Ok(scored) = scored {
                self.update(candidate, scored);
            }
        }
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
