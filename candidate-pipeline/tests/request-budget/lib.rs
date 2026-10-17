// Public-source harness: use actual pipeline code and public async/tracing crates.
// Only unpublished feature/decider/metrics adapters are stand-ins. These tests do
// not validate production telemetry, feature-switch behavior, or service wiring.
#![allow(dead_code)]
extern crate self as xai_decider;
extern crate self as xai_feature_switches;
extern crate self as xai_stats_receiver;

#[derive(Clone, Default)]
pub struct Params;
pub struct Decider;
pub struct StatsReceiver;
pub enum HistogramBuckets {
    Bucket0To50,
}
impl StatsReceiver {
    pub fn incr(&self, _: &str, _: &[(&str, &str)], _: u64) {}
    pub fn observe(&self, _: &str, _: &[(&str, &str)], _: f64, _: HistogramBuckets) {}
}
pub fn global_stats_receiver() -> Option<&'static StatsReceiver> {
    None
}
pub async fn with_scoped_metric_labels<F: std::future::Future>(
    _: Vec<(String, String)>,
    future: F,
) -> F::Output {
    future.await
}
pub const SPAN_LEVEL: tracing::Level = tracing::Level::INFO;

#[path = "../../candidate_pipeline.rs"]
pub mod candidate_pipeline;
#[path = "../../filter.rs"]
pub mod filter;
#[path = "../../hydrator.rs"]
pub mod hydrator;
#[path = "../../pipeline_summary.rs"]
pub mod pipeline_summary;
#[path = "../../query_hydrator.rs"]
pub mod query_hydrator;
#[path = "../../scorer.rs"]
pub mod scorer;
#[path = "../../selector.rs"]
pub mod selector;
#[path = "../../side_effect.rs"]
pub mod side_effect;
#[path = "../../source.rs"]
pub mod source;
#[path = "../../util.rs"]
pub mod util;

#[cfg(test)]
mod tests;
