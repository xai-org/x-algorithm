use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge, register_histogram_vec, CounterVec, Gauge, Histogram,
    HistogramOpts, HistogramVec, Opts,
};

lazy_static! {
    pub static ref SCORE_REQUESTS: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_requests_total",
            "Total number of Rank RPC requests"
        ),
        &["value_model_id"]
    )
    .unwrap();
    pub static ref SCORE_ERRORS: CounterVec = register_counter_vec!(
        Opts::new("vm_ranker_errors_total", "Total number of Rank RPC errors"),
        &["value_model_id"]
    )
    .unwrap();
    pub static ref SCORE_DURATION: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_duration_seconds",
            "End-to-end Rank RPC latency in seconds"
        )
        .buckets(prometheus::exponential_buckets(0.001, 1.3, 30).unwrap()),
        &["value_model_id"]
    )
    .unwrap();
    pub static ref SCORE_CANDIDATES_IN: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_candidates_in",
            "Number of candidates received per Rank request"
        )
        .buckets(prometheus::exponential_buckets(1.0, 1.2, 55).unwrap()),
        &["value_model_id"]
    )
    .unwrap();
    pub static ref IN_FLIGHT_REQUESTS: Gauge = register_gauge!(
        "vm_ranker_in_flight_requests",
        "Number of Rank requests currently being processed"
    )
    .unwrap();
    pub static ref REJECTED_REQUESTS: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_rejected_requests_total",
            "Total number of requests rejected due to concurrency limit (RESOURCE_EXHAUSTED)"
        ),
        &["value_model_id"]
    )
    .unwrap();
    pub static ref DPP_RESCALING: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_rescaling",
            "Per-request count of DPP-selected candidates by rescaling direction"
        )
        .buckets(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0
        ]),
        &["direction"]
    )
    .unwrap();
    pub static ref DPP_TOP_K_OVERLAP: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_top_k_overlap",
            "Overlap count between DPP-selected items and original top-K (by input score)"
        )
        .buckets(vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0
        ]),
        &["k"]
    )
    .unwrap();
    pub static ref DPP_SELECTED_COUNT: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_selected_count",
            "Number of items selected by greedy DPP per request"
        )
        .buckets(vec![
            0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 55.0,
            75.0, 100.0
        ]),
        &["top_k"]
    )
    .unwrap();
    pub static ref DPP_LOG_DET: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_log_det",
            "Log-determinant of the DPP-selected subset (measures diversity volume)"
        )
        .buckets(prometheus::exponential_buckets(0.01, 2.0, 30).unwrap()),
        &["top_k"]
    )
    .unwrap();
    pub static ref DPP_AVG_SIMILARITY: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_avg_similarity",
            "Average pairwise cosine similarity before/after DPP (lower = more diverse)"
        )
        .buckets(vec![
            -1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
            0.7, 1.0
        ]),
        &["stage"]
    )
    .unwrap();
    pub static ref DPP_TERMINAL_CV: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_terminal_cv",
            "Conditional variance of the best unchosen item when DPP stopped selecting"
        )
        .buckets(vec![
            1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0,
        ]),
        &["reason"]
    )
    .unwrap();
    pub static ref DPP_POOL_SIZE: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_pool_size",
            "Number of candidates in the DPP pool after pre-filtering"
        )
        .buckets(vec![
            0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0, 300.0
        ]),
        &["max_selected_rank"]
    )
    .unwrap();
    pub static ref DPP_POOL_SCORES: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_pool_scores",
            "Score distribution summary of candidates in the DPP pool"
        )
        .buckets(vec![
            -10.0, -5.0, -1.0, -0.5, -0.1, 0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
            50.0, 100.0, 500.0, 1000.0,
        ]),
        &["stat"]
    )
    .unwrap();
    pub static ref DPP_EMBEDDING_MISS_RATIO: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_embedding_miss_ratio",
            "Fraction of DPP candidate pool items using deterministic fallback embeddings"
        )
        .buckets(vec![
            0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0
        ]),
        &["pool_size"]
    )
    .unwrap();
}

pub struct Timer {
    histogram: Histogram,
    start: std::time::Instant,
}

impl Timer {
    pub fn new(histogram: Histogram) -> Self {
        Self {
            histogram,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}
