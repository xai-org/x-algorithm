use std::any::{type_name_of_val, Any};
use tonic::async_trait;

use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use crate::SPAN_LEVEL;
use tracing::error;

#[async_trait]
pub trait Source<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(size=Bucket500To1000)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "source", fields(name = self.name()))]
    async fn run(&self, query: &Q, stage: PipelineStage) -> Result<Vec<C>, String> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        match self.source(query).await {
            Ok(candidates) => {
                stats.finish_source(candidates.len());
                Ok(candidates)
            }
            Err(err) => {
                stats.finish_failed();
                error!("{} Failed: {}", self.name(), err);
                Err(err)
            }
        }
    }

    async fn source(&self, query: &Q) -> Result<Vec<C>, String>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
