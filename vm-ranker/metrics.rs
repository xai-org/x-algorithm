use lazy_static::lazy_static;
use prometheus::{
    register_counter_vec, register_gauge, register_histogram, register_histogram_vec,
    register_int_counter, register_int_counter_vec, register_int_gauge, register_int_gauge_vec,
    CounterVec, Gauge, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge,
    IntGaugeVec, Opts,
};

lazy_static! {
    pub static ref VALUE_MODEL_REQUESTS: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_value_model_requests_total",
            "Rank requests by scoring mode and whether user debias was applied"
        ),
        &["mode", "debias"]
    )
    .unwrap();
    pub static ref VALUE_MODEL_FALLBACK: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_value_model_fallback_total",
            "Rank requests that fell back to the upstream candidate score"
        ),
        &["reason"]
    )
    .unwrap();
    pub static ref VALUE_MODEL_STAGE: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_value_model_stage_total",
            "Value-model requests by whether heads were weighted (heads) or the caller supplied every candidate's weighted score and only per-request adjustments ran (cached_weighted)"
        ),
        &["stage"]
    )
    .unwrap();
    pub static ref USER_DEBIAS_REQUESTS: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_user_debias_requests_total",
            "User-debias requests by outcome: applied with Phoenix scores from this request (fresh), from the post cache (cached) or without a scoring time (no_hint); skipped because of model_not_loaded, stale_artifact, no_sequence, invalid_params, sequence_error or feature_mismatch"
        ),
        &["path"]
    )
    .unwrap();
    pub static ref USER_DEBIAS_PREDICT_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "vm_ranker_user_debias_predict_seconds",
            "Sequence parse + featurize + predict latency in seconds"
        )
        .buckets(prometheus::exponential_buckets(0.0001, 1.5, 25).unwrap())
    )
    .unwrap();
    pub static ref USER_DEBIAS_MODEL_LOADED: IntGauge = register_int_gauge!(
        "vm_ranker_user_debias_model_loaded",
        "1 when a user-debias model is loaded, else 0"
    )
    .unwrap();
    pub static ref USER_DEBIAS_MODEL_TRAINED_AT_MS: IntGauge = register_int_gauge!(
        "vm_ranker_user_debias_model_trained_at_ms",
        "trained_at_ms of the loaded user-debias model"
    )
    .unwrap();
    pub static ref USER_DEBIAS_MODEL_TRAIN_RECORDS: IntGauge = register_int_gauge!(
        "vm_ranker_user_debias_model_train_records",
        "train_records of the loaded user-debias model"
    )
    .unwrap();
    pub static ref USER_DEBIAS_MODEL_RELOADS: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_user_debias_model_reloads_total",
            "User-debias model reload attempts by outcome"
        ),
        &["outcome"]
    )
    .unwrap();
    pub static ref CAROUSEL_MODELS_LOADED: IntGauge = register_int_gauge!(
        "vm_ranker_carousel_models_loaded",
        "Number of video carousel model files held in memory"
    )
    .unwrap();
    pub static ref CAROUSEL_MODEL_RELOADS: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_carousel_model_reloads_total",
            "Video carousel model directory reads by outcome: ok, dir_error (kept all previous files) or file_error (kept that file's previous copy)"
        ),
        &["outcome"]
    )
    .unwrap();
    pub static ref CAROUSEL_MODEL_REQUESTS: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_carousel_model_requests_total",
            "GetCarouselModel requests by outcome: ok, not_found or disabled (no --carousel-model-dir)"
        ),
        &["outcome"]
    )
    .unwrap();
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
            pub static ref DPP_SEED_CONTEXT: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_dpp_seed_context_total",
            "Rank requests with a seed_tweet_id, by whether the seed embedding was found"
        ),
        &["result"]
    )
    .unwrap();
            pub static ref DPP_RESCALING: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_rescaling",
            "Per-request count of DPP-selected candidates by rescaling direction"
        )
        .buckets(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0]),
        &["direction"]
    )
    .unwrap();
        pub static ref DPP_TOP_K_OVERLAP: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_top_k_overlap",
            "Overlap count between DPP-selected items and original top-K (by input score)"
        )
        .buckets(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0]),
        &["k"]
    )
    .unwrap();
            pub static ref DPP_SELECTED_COUNT: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_selected_count",
            "Number of items selected by greedy DPP per request"
        )
        .buckets(vec![0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 55.0, 75.0, 100.0]),
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
        .buckets(vec![-1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]),
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
        .buckets(vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0, 300.0]),
        &["max_selected_rank"]
    )
    .unwrap();
                    pub static ref DPP_POOL_SCORES: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_pool_scores",
            "Score distribution summary of candidates in the DPP pool"
        )
        .buckets(vec![
            -10.0, -5.0, -1.0, -0.5, -0.1, 0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0,
            10.0, 20.0, 50.0, 100.0, 500.0, 1000.0,
        ]),
        &["stat"]
    )
    .unwrap();
            pub static ref DPP_EMBEDDING_MISS_RATIO: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_dpp_embedding_miss_ratio",
            "Fraction of DPP candidate pool items with missing embeddings (zero vector)"
        )
        .buckets(vec![0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]),
        &["pool_size"]
    )
    .unwrap();
    pub static ref CONFIG_SYNC_LAST_SUCCESS_SECS: Gauge = register_gauge!(
        "vm_ranker_config_sync_last_success_timestamp_seconds",
        "Unix time of the last successful config poll (fetch succeeded, files current)"
    )
    .unwrap();
    pub static ref CONFIG_SYNC_HEAD_AGE_SECS: Gauge = register_gauge!(
        "vm_ranker_config_sync_head_age_seconds",
        "Age of the served config revision's commit at the last successful poll"
    )
    .unwrap();
    pub static ref CONFIG_SYNC_FAILURES: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_config_sync_failures_total",
            "Config poll failures by stage"
        ),
        &["stage"]
    )
    .unwrap();
    pub static ref CONFIG_SYNC_APPLIED: IntCounter = register_int_counter!(
        "vm_ranker_config_sync_applied_total",
        "Config revisions applied (files replaced and feature switches reloaded)"
    )
    .unwrap();
    pub static ref CONFIG_SYNC_REVISION: IntGaugeVec = register_int_gauge_vec!(
        Opts::new(
            "vm_ranker_config_sync_revision",
            "1 for the config-repo revision currently served"
        ),
        &["revision"]
    )
    .unwrap();
    pub static ref PARAMS_RESOLVED: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_params_resolved_total",
            "Rank requests by whether ranking parameters were resolved from the viewer context"
        ),
        &["outcome"]
    )
    .unwrap();
    pub static ref AUTHOR_EXPLORATION_CANDIDATES: IntCounterVec = register_int_counter_vec!(
        Opts::new(
            "vm_ranker_author_exploration_candidates_total",
            "Value-model candidates by whether their author has a non-zero exploration bonus"
        ),
        &["bonus"]
    )
    .unwrap();
    pub static ref SAMPLED_CANDIDATES: IntCounter = register_int_counter!(
        "vm_ranker_sampled_candidates_total",
        "Candidates with Phoenix heads in sampled value-model requests"
    )
    .unwrap();
    pub static ref HEAD_PREDICTION_SUM: CounterVec = register_counter_vec!(
        Opts::new(
            "vm_ranker_head_prediction_sum",
            "Sum of Phoenix head predictions over sampled candidates"
        ),
        &["head"]
    )
    .unwrap();
    pub static ref CANDIDATE_SCORE: HistogramVec = register_histogram_vec!(
        HistogramOpts::new(
            "vm_ranker_candidate_score",
            "Per-candidate value-model scores in sampled requests: weighted (offset base) and ranked (after cold start, author diversity and OON, before DPP)"
        )
        .buckets(vec![
            0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,
            0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0,
        ]),
        &["stage"]
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
