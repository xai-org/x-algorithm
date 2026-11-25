// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
pub mod arrow;
mod error;
pub mod features;
pub mod model;

pub use arrow::parse_sequence;
pub use error::Error;
pub use features::{
    FEATURE_DIM, FEATURE_VERSION, HISTORY_STATS_DIM, RequestContext, SequenceColumns, SequenceRows,
    featurize_batch, featurize_sequence, history_stats, history_stats_batch, request_context,
    surface_id,
};
pub use model::{ArtifactTensor, Prediction, UserDebiasModel, write_artifact};

pub fn debiased_probability(p: f64, user_logit: f32, global_logit: f32, clip: f64) -> f64 {
    let delta = (user_logit as f64 - global_logit as f64).clamp(-clip, clip);
    let z = (p / (1.0 - p)).ln() - delta;
    1.0 / (1.0 + (-z).exp())
}

pub fn debiased_dwell(d: f64, user_log_dwell: f32, global_log_dwell: f32, clip: f64) -> f64 {
    let delta = (user_log_dwell as f64 - global_log_dwell as f64).clamp(-clip, clip);
    d * (-delta).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_stats_read_only_the_newest_rows_a_training_record_holds() {
        let max_rows = features::HISTORY_STATS_MAX_ROWS;
        let n = max_rows + 5;
        let rows = SequenceRows {
            acts: (0..n).map(|i| 1u64 << (i % 64)).collect(),
            ts_ms: (0..n as i64)
                .map(|i| 1_700_000_000_000 + i * 3_600_000)
                .collect(),
            surface: (0..n).map(|i| if i % 2 == 0 { 10 } else { 1 }).collect(),
            tz: vec![1; n],
            dwell_sec: (0..n).map(|i| (i % 30) as f32).collect(),
            has_media: vec![true; n],
            media_type: vec![1; n],
            video_dur_ms: vec![30_000; n],
            in_network: vec![false; n],
            position: (0..n as i32).collect(),
        };
        let req_ms = rows.ts_ms[n - 1] + 1_000;
        let mut newest = rows.clone();
        newest.retain(&(0..n).map(|i| i >= n - max_rows).collect::<Vec<_>>());
        assert_ne!(
            featurize_sequence(&rows, req_ms),
            featurize_sequence(&newest, req_ms)
        );
        assert_eq!(history_stats(&rows, req_ms), history_stats(&newest, req_ms));
    }
}
