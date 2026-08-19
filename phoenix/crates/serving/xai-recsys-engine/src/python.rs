// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::adler32::{
    adler32_combine_partials, adler32_combine_py, adler32_parallel, adler32_shard_partial,
};
#[cfg(target_os = "linux")]
use crate::copy::{CopyService, Sender};
#[cfg(target_os = "linux")]
use crate::copy::{DEFAULT_MAX_ENTRIES, ManifestSpec};
#[cfg(not(target_os = "linux"))]
const DEFAULT_MAX_ENTRIES: usize = 12;
use crate::emb_table::{
    embedding_gather, init_parallel, load_tensor, load_tensor_no_resharding, madvise_hugepage,
    madvise_normal, madvise_sequential, maybe_load_fully_replicated_tensors,
};
use crate::multimodal_retrieval::PrefetchMmQueryConfig;
use crate::request_metrics::{
    self, ADMISSION_BUDGET_REMAINING, ADMISSION_PREDICTED_ETA, BATCHING_TIMEOUT_DROP,
    DEADLINE_SHED, ITEMS_IN_FLIGHT as IN_FLIGHT, NUM_REQUESTS_CANCELED, NUM_REQUESTS_FAILED,
    NUM_REQUESTS_RECEIVED, NUM_REQUESTS_REJECTED, NUM_REQUESTS_STALE_DROPPED,
    NUM_REQUESTS_SUCCEEDED, QUEUE_BACKLOG, SEQUENCES_PROCESSED, TOKENS_PADDED, TOKENS_PER_ROW,
    TOKENS_PROCESSED, arm_not_ready_probe, client_label, disarm_not_ready_probe,
    extract_grpc_timeout, record_client_server_network_delay, record_grpc_timeout_presence,
    record_not_ready_arrival, remaining_deadline_budget_us, src_dc_label, status_code,
    workload_label,
};
use crate::request_queue::RequestQueue;
use crate::storage_util::{
    copy_chunks_to_storage, download_files_from_storage, download_prefix_to_local,
    find_latest_storage_checkpoint, list_storage_objects, load_arrays_from_storage,
    load_reshard_row_to_col, remove_old_from_storage, reshard_col_to_row_and_upload,
    upload_files_to_storage, upload_inline_data_to_storage,
};
use crate::util::RetrievalInputBuffer;
use crate::util::{PredictRequestItem, RetrieveRequestItem};
use autometrics::prometheus_exporter;

const BLOOM_NUM_HASHES: usize = 6;

const NUM_CONVERSION_DENSE_FEATURES: usize = 29;
const NUM_CONVERSION_HISTORY_HASHES: usize = 8;

#[inline]
fn bloom_may_contain(bit_array: &[u64], num_bits: usize, post_id: i64) -> bool {
    let h128 =
        murmur3::murmur3_x64_128(&mut std::io::Cursor::new(&post_id.to_le_bytes()), 0).unwrap_or(0);
    let h1 = h128 as u64;
    let h2 = (h128 >> 64) as u64;
    (0..BLOOM_NUM_HASHES).all(|i| {
        let idx = (h1.wrapping_add((i as u64).wrapping_mul(h2)) % num_bits as u64) as usize;
        idx < num_bits && (bit_array[idx >> 6] & (1u64 << (idx & 63))) != 0
    })
}
use axum::routing::get;
use bytes::Bytes;
use chrono::TimeDelta;
use half::f16;
use lazy_static::lazy_static;
use log::warn;
use numpy::{
    PyArray1, PyArray2, PyArray3, ToPyArray,
    ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, s},
};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use prometheus::{
    Histogram, IntGauge, exponential_buckets, register_histogram, register_int_gauge,
};
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{Bound, Py, Python};
use rayon;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_util::sync::DropGuard;
use tonic::{Request, Response, Status, service::RoutesBuilder};
use xai_recsys::{model_config::ModelConfig, util::InputBuffer};
use xai_recsys_server::LoadGauge;
use xai_recsys_server::{CancellationToken, GrpcConfig, HttpServer};

use crate::admission::{AdmissionConfig, AdmissionController, AdmissionEtaModel};
use xai_recsys_mm_server::PyMmEmbeddingsClient;
use xai_recsys_mm_server::mm_embedding_client::MmEmbeddingsClient;
use xai_recsys_proto::{
    self as pb, CandidateDistributionSet, CandidateSet, ScoredCandidates, UserActionSequence,
};

struct InFlightGuard;

impl InFlightGuard {
    fn new() -> Self {
        IN_FLIGHT.add(1);
        Self
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        IN_FLIGHT.sub(1);
    }
}

lazy_static! {
    static ref IN_PROGRESS: IntGauge = register_int_gauge!(
        "recsys_engine_items_in_progress",
        "Number of recsys items that are in progress / being sampled"
    )
    .unwrap();
    static ref REQUEST_LATENCY: Histogram = {
        let buckets = generate_latency_buckets();
        register_histogram!(
            "recsys_engine_request_latency",
            "Measures total latency to process a request",
            buckets
        )
        .unwrap()
    };
    static ref INPUT_BUFFER_COMPUTATION_TIME: Histogram = register_histogram!(
        "recsys_engine_input_buffer_computation_time_seconds",
        "Time taken to compute InputBuffer for individual requests",
        exponential_buckets(0.00001, 1.177, 100).unwrap()
    )
    .unwrap();
    static ref GET_CANDIDATE_SETS_LATENCY: Histogram = register_histogram!(
        "recsys_engine_get_candidate_sets_latency_seconds",
        "Time taken for get_candidate_sets",
        exponential_buckets(0.000001, 1.177, 100).unwrap()
    )
    .unwrap();
    static ref PREDICT_REPLY_LATENCY: Histogram = register_histogram!(
        "recsys_engine_predict_reply_latency_seconds",
        "Time taken to process and send predict reply responses",
        exponential_buckets(0.0001, 1.5, 30).unwrap()
    )
    .unwrap();
    static ref RETRIEVE_REPLY_LATENCY: Histogram = register_histogram!(
        "recsys_engine_retrieve_reply_latency_seconds",
        "Time taken to process and send retrieve reply responses",
        exponential_buckets(0.0001, 1.5, 30).unwrap()
    )
    .unwrap();
}

fn generate_latency_buckets() -> Vec<f64> {
    request_metrics::latency_buckets_ms()
}

fn extract_request_id<T>(request: &Request<T>) -> String {
    if let Some(request_id) = request.metadata().get("x-request-id")
        && let Ok(id_str) = request_id.to_str()
    {
        return id_str.to_string();
    }

    if let Some(request_id) = request.metadata().get("request-id")
        && let Ok(id_str) = request_id.to_str()
    {
        return id_str.to_string();
    }

    "default".to_string()
}

fn queue_backlog<T>(sender: &mpsc::Sender<T>) -> i64 {
    sender.max_capacity().saturating_sub(sender.capacity()) as i64
        + request_metrics::WAITING_QUEUE_SIZE.get()
}

fn should_shed_for_deadline<T>(
    admission: &AdmissionController,
    sender: &mpsc::Sender<T>,
    grpc_timeout: Option<Duration>,
    elapsed_us: u64,
) -> Option<String> {
    if !admission.is_enabled() {
        return None;
    }
    let backlog = queue_backlog(sender);
    let eta_us = admission.predicted_eta_us(backlog);
    if let Some(eta_us) = eta_us {
        ADMISSION_PREDICTED_ETA.observe(eta_us as f64 / 1000.0);
    }
    let remaining_us =
        remaining_deadline_budget_us(grpc_timeout, admission.fallback_budget_us(), elapsed_us);
    if !admission.should_reject(remaining_us, backlog) {
        return None;
    }
    let budget_source = if grpc_timeout.is_some() {
        "grpc-timeout"
    } else {
        "deadline_fallback_budget_ms"
    };
    Some(format!(
        "shed at admission: predicted completion {}ms + margin {}ms > remaining {}ms (from {budget_source}), backlog={backlog}",
        eta_us.unwrap_or(0) / 1000,
        admission.margin_us() / 1000,
        remaining_us.unwrap_or(0) / 1000,
    ))
}

const MAX_MESSAGE_SIZE: usize = 128 * 1024 * 1024;
const CHANNEL_SIZE: usize = 2048;
const ENQUEUE_TIMEOUT_MS: u64 = 1000;
const QUEUE_MAX_STALENESS_MS: u64 = 1200;

#[pyclass(module = "xai_recsys_engine", get_all)]
pub struct ReqCandidateSequence {
    author_ids: Py<PyArray2<i64>>,
    tweet_ids: Py<PyArray2<i64>>,
    can_lengths: Py<PyArray1<i64>>,
}

#[pyclass(module = "xai_recsys_engine", get_all)]
pub struct ReqUserSequenceAggrgated {
    user_ids: Py<PyArray1<i64>>,
    post_lengths: Py<PyArray1<i64>>,
    author_ids: Py<PyArray2<i64>>,
    tweet_ids: Py<PyArray2<i64>>,
    actions: Py<PyArray3<bool>>,
    impressed_time_ms: Py<PyArray2<i64>>,
}

pub trait RequestItem {
    fn get_sequence(&self) -> &Option<pb::UserActionSequence>;
    fn get_user_id(&self) -> i64;
}

impl RequestItem for PredictRequestItem {
    fn get_sequence(&self) -> &Option<pb::UserActionSequence> {
        &self.sequence
    }

    fn get_user_id(&self) -> i64 {
        self.candidate_set.user_id as i64
    }
}

impl RequestItem for RetrieveRequestItem {
    fn get_sequence(&self) -> &Option<pb::UserActionSequence> {
        &self.sequence
    }

    fn get_user_id(&self) -> i64 {
        self.user_id as i64
    }
}

fn stale_drop_message(
    enqueue_time: Instant,
    deadline: Option<Instant>,
    max_staleness: Duration,
) -> String {
    let queued_ms = enqueue_time.elapsed().as_millis();
    match deadline {
        Some(d) => {
            let overdue_ms = Instant::now().saturating_duration_since(d).as_millis();
            format!(
                "Request dropped: queued {queued_ms}ms, client grpc-timeout deadline exceeded by {overdue_ms}ms"
            )
        }
        None => format!(
            "Request dropped: queued {queued_ms}ms > queue_max_staleness_ms={}",
            max_staleness.as_millis()
        ),
    }
}

impl crate::request_queue::QueuedItem for PredictRequestItem {
    fn enqueue_time(&self) -> Instant {
        self.enqueue_time
    }

    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    fn on_stale_drop(mut self, max_staleness: Duration) {
        NUM_REQUESTS_STALE_DROPPED.inc();
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err(Status::deadline_exceeded(stale_drop_message(
                self.enqueue_time,
                self.deadline,
                max_staleness,
            ))));
        }
    }
}

impl crate::request_queue::QueuedItem for RetrieveRequestItem {
    fn enqueue_time(&self) -> Instant {
        self.enqueue_time
    }

    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    fn on_stale_drop(mut self, max_staleness: Duration) {
        NUM_REQUESTS_STALE_DROPPED.inc();
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err(Status::deadline_exceeded(stale_drop_message(
                self.enqueue_time,
                self.deadline,
                max_staleness,
            ))));
        }
    }
}

#[pyclass(module = "xai_recsys_engine")]
pub struct PredictRequestBatch {
    items: Vec<PredictRequestItem>,
}

#[pymethods]
impl PredictRequestBatch {
    pub fn get_candidate_sets<'py>(
        &self,
        py: Python<'py>,
        max_num_cands: usize,
    ) -> PyResult<ReqCandidateSequence> {
        let start = Instant::now();
        let b = self.items.len();

        let mut author_ids = Array2::<i64>::zeros((b, max_num_cands));
        let mut tweet_ids = Array2::<i64>::zeros((b, max_num_cands));
        let mut can_lengths = Array1::<i64>::zeros(b);

        let mut authors = Vec::with_capacity(max_num_cands);
        let mut tweets = Vec::with_capacity(max_num_cands);

        for (i, item) in self.items.iter().enumerate() {
            let candidates = &item.candidate_set.candidates
                [..max_num_cands.min(item.candidate_set.candidates.len())];

            authors.clear();
            tweets.clear();

            for candidate in candidates.iter() {
                authors.push(candidate.author_id as i64);
                tweets.push(candidate.tweet_id as i64);
                can_lengths[i] += 1;
            }

            let l = authors.len().min(max_num_cands);
            if l > 0 {
                author_ids
                    .slice_mut(s![i, ..l])
                    .assign(&ArrayView1::from(&authors[..l]));
                tweet_ids
                    .slice_mut(s![i, ..l])
                    .assign(&ArrayView1::from(&tweets[..l]));
            }
        }
        GET_CANDIDATE_SETS_LATENCY.observe(start.elapsed().as_secs_f64());
        Ok(ReqCandidateSequence {
            author_ids: author_ids.to_pyarray(py).into(),
            tweet_ids: tweet_ids.to_pyarray(py).into(),
            can_lengths: can_lengths.to_pyarray(py).into(),
        })
    }

    pub fn get_return_logprob<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        let return_logprobs: Vec<bool> =
            self.items.iter().map(|item| item.return_logprob).collect();
        PyArray2::from_vec2(py, &[return_logprobs]).unwrap()
    }

    pub fn get_top_logprobs_num<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let top_logprobs_nums: Vec<u32> = self
            .items
            .iter()
            .map(|item| item.top_logprobs_num)
            .collect();
        PyArray2::from_vec2(py, &[top_logprobs_nums]).unwrap()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn reply(
        &mut self,
        py: Python<'_>,
        responses: Bound<'_, PyArray3<f32>>,
        continuous_predictions: Bound<'_, PyArray3<f32>>,
        batch_size: usize,
    ) -> PyResult<()> {
        let responses = DonatedArray3::new(responses)?;
        let continuous_predictions = DonatedArray3::new(continuous_predictions)?;

        let items = std::mem::take(&mut self.items);

        py.detach(|| {
            thread::spawn(move || {
                let start = Instant::now();

                let responses = unsafe { responses.as_array_view() };
                let continuous_predictions = unsafe { continuous_predictions.as_array_view() };

                for (i, ((mut item, response), cont_pred_batch)) in items
                    .into_iter()
                    .zip(responses.outer_iter())
                    .zip(continuous_predictions.outer_iter())
                    .enumerate()
                {
                    if i >= batch_size {
                        break;
                    }
                    let candidate_distributions = response
                        .outer_iter()
                        .zip(item.candidate_set.candidates.iter())
                        .zip(cont_pred_batch.outer_iter())
                        .map(|((dist, candidate), cont_pred)| {
                            let mut dist_entry = pb::NextActionDistribution::default();

                            if item.return_candidate_tweet_id_only {
                                dist_entry.candidate = Some(pb::TweetInfo {
                                    tweet_id: candidate.tweet_id,
                                    ..Default::default()
                                });
                            } else {
                                dist_entry.candidate = Some(candidate.clone());
                            }

                            if item.return_log_map {
                                for idx in item.requested_action_indices.iter() {
                                    if let Some(v) = dist.get(*idx as usize) {
                                        dist_entry.index_to_logits.insert(*idx, *v);
                                    }
                                }
                                for idx in item.requested_continuous_action_indices.iter() {
                                    if let Some(v) = cont_pred.get(*idx as usize) {
                                        dist_entry.index_to_continuous_values.insert(*idx, *v);
                                    }
                                }
                            } else if item.return_logprob {
                                dist_entry.top_log_probs = dist.to_vec();
                                dist_entry.continuous_actions_values = cont_pred.to_vec();
                            } else {
                                dist_entry.logits = dist.to_vec();
                                dist_entry.continuous_actions_values = cont_pred.to_vec();
                            }

                            dist_entry
                        })
                        .collect();

                    let distribution_set = pb::CandidateDistributionSet {
                        user_id: item.candidate_set.user_id,
                        candidate_distributions,
                        ..Default::default()
                    };

                    if let Some(sender) = item.sender.take() {
                        sender.send(Ok(distribution_set)).ok();
                    }
                }
                let duration = start.elapsed();
                PREDICT_REPLY_LATENCY.observe(duration.as_secs_f64());
                log::debug!("PredictRequestBatch::reply took {:?}", duration);
            });
        });

        Ok(())
    }
}

struct DonatedArray1<T> {
    ptr: *const T,
    len: usize,
    _guard: Py<PyArray1<T>>,
}

struct DonatedArray2<T> {
    ptr: *const T,
    rows: usize,
    cols: usize,
    _guard: Py<PyArray2<T>>,
}

struct DonatedArray3<T> {
    ptr: *const T,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    _guard: Py<PyArray3<T>>,
}

unsafe impl<T: Send> Send for DonatedArray1<T> {}
unsafe impl<T: Send> Send for DonatedArray2<T> {}
unsafe impl<T: Send> Send for DonatedArray3<T> {}

impl<T: numpy::Element> DonatedArray1<T> {
    fn new(arr: Bound<'_, PyArray1<T>>) -> PyResult<Self> {
        if !arr.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Array must be C-contiguous",
            ));
        }
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let ptr = view.as_ptr();
        let len = view.len();
        let guard = arr.unbind();
        Ok(Self {
            ptr,
            len,
            _guard: guard,
        })
    }

    #[inline]
    unsafe fn get(&self, idx: usize) -> T
    where
        T: Copy,
    {
        debug_assert!(idx < self.len, "index out of bounds");
        unsafe { *self.ptr.add(idx) }
    }
}

impl<T: numpy::Element> DonatedArray2<T> {
    fn new(arr: Bound<'_, PyArray2<T>>) -> PyResult<Self> {
        if !arr.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Array must be C-contiguous",
            ));
        }
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let ptr = view.as_ptr();
        let shape = view.shape();
        let rows = shape[0];
        let cols = shape[1];
        let guard = arr.unbind();
        Ok(Self {
            ptr,
            rows,
            cols,
            _guard: guard,
        })
    }

    #[inline]
    unsafe fn as_array_view(&self) -> ArrayView2<'_, T> {
        unsafe { ArrayView2::from_shape_ptr((self.rows, self.cols), self.ptr) }
    }
}

impl<T: numpy::Element> DonatedArray3<T> {
    fn new(arr: Bound<'_, PyArray3<T>>) -> PyResult<Self> {
        if !arr.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Array must be C-contiguous",
            ));
        }
        let readonly = arr.readonly();
        let view = readonly.as_array();
        let ptr = view.as_ptr();
        let shape = view.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];
        let guard = arr.unbind();
        Ok(Self {
            ptr,
            dim0,
            dim1,
            dim2,
            _guard: guard,
        })
    }

    #[inline]
    unsafe fn as_array_view(&self) -> ArrayView3<'_, T> {
        unsafe { ArrayView3::from_shape_ptr((self.dim0, self.dim1, self.dim2), self.ptr) }
    }
}

#[pyclass(module = "xai_recsys_engine")]
struct RetrieveRequestBatch {
    items: Vec<RetrieveRequestItem>,
}

#[pymethods]
impl RetrieveRequestBatch {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get_top_ks(&self) -> Vec<u32> {
        self.items.iter().map(|item| item.top_k).collect()
    }

    pub fn get_topic_entity_ids(&self) -> Vec<Vec<u64>> {
        self.items
            .iter()
            .map(|item| item.topic_entity_ids.clone())
            .collect()
    }

    pub fn get_topic_filter_mode(&self) -> i32 {
        self.items
            .first()
            .map(|item| item.topic_filter_mode)
            .unwrap_or(0)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_pre_fetched_seed_embeddings<'py>(
        &self,
        py: Python<'py>,
        emb_dim: usize,
    ) -> PyResult<Vec<(Vec<u64>, Bound<'py, PyArray2<f32>>)>> {
        crate::multimodal_retrieval::read_pre_fetched_seed_embeddings(&self.items, py, emb_dim)
    }

    pub fn build_eligible_mask<'py>(
        &self,
        py: Python<'py>,
        all_post_ids: &Bound<'py, PyArray1<i64>>,
        batch_size: usize,
    ) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        if self
            .items
            .iter()
            .all(|item| item.eligible_posts_bloom_filter.is_empty())
        {
            return Ok(None);
        }

        let post_ids_ro = unsafe { all_post_ids.as_slice()? };
        let m = post_ids_ro.len();
        let mut mask = Array2::<bool>::from_elem((batch_size, m), true);

        for (user_idx, item) in self.items.iter().enumerate() {
            if user_idx >= batch_size {
                break;
            }
            let bf = &item.eligible_posts_bloom_filter;
            if bf.is_empty() {
                continue;
            }
            let num_bits = bf.len() * 8;
            let bit_array: &[u64] = match bytemuck::try_cast_slice(bf) {
                Ok(arr) => arr,
                Err(_) => continue,
            };

            for j in 0..m {
                let pid = post_ids_ro[j];
                if pid == 0 {
                    mask[[user_idx, j]] = false;
                    continue;
                }
                mask[[user_idx, j]] = bloom_may_contain(bit_array, num_bits, pid);
            }
        }

        Ok(Some(mask.to_pyarray(py)))
    }

    const ACTIVE_ADS_DS_TYPE: i32 = 7;

    pub fn build_eligible_mask_into_shards(
        &self,
        py: Python<'_>,
        all_post_ids: &Bound<'_, PyArray1<i64>>,
        dataset_types: &Bound<'_, PyArray1<i32>>,
        batch_size: usize,
        shard_buffers: Vec<Bound<'_, PyArray1<u8>>>,
    ) -> PyResult<()> {
        let post_ids_ro = unsafe { all_post_ids.as_slice()? };
        let ds_types_ro = unsafe { dataset_types.as_slice()? };
        let m = post_ids_ro.len();
        let num_shards = shard_buffers.len();
        let users_per_shard = batch_size / num_shards;

        let shard_ptrs: Vec<(*mut u8, usize)> = shard_buffers
            .iter()
            .map(|buf| {
                let slice = unsafe { buf.as_slice_mut() }.expect("shard buffer not contiguous");
                (slice.as_mut_ptr(), slice.len())
            })
            .collect();

        let has_bloom = self
            .items
            .iter()
            .take(batch_size)
            .any(|item| !item.eligible_posts_bloom_filter.is_empty());

        if !has_bloom {
            for &(ptr, len) in &shard_ptrs {
                unsafe { std::ptr::write_bytes(ptr, 1u8, len) };
            }
            return Ok(());
        }

        struct SendPtr(*mut u8, usize);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let send_ptrs: Vec<SendPtr> = shard_ptrs
            .iter()
            .map(|&(ptr, len)| SendPtr(ptr, len))
            .collect();

        let items_ref = &self.items;
        py.detach(|| {
            std::thread::scope(|s| {
                for (shard_idx, sp) in send_ptrs.iter().enumerate() {
                    s.spawn(move || {
                        let buf = unsafe { std::slice::from_raw_parts_mut(sp.0, sp.1) };
                        for local_user in 0..users_per_shard {
                            let user_idx = shard_idx * users_per_shard + local_user;
                            let row_start = local_user * m;

                            if user_idx >= items_ref.len() || user_idx >= batch_size {
                                buf[row_start..row_start + m].fill(1);
                                continue;
                            }

                            let bf = &items_ref[user_idx].eligible_posts_bloom_filter;
                            if bf.is_empty() {
                                buf[row_start..row_start + m].fill(1);
                                continue;
                            }

                            let num_bits = bf.len() * 8;
                            let bit_array: &[u64] = match bytemuck::try_cast_slice(bf) {
                                Ok(arr) => arr,
                                Err(_) => {
                                    buf[row_start..row_start + m].fill(1);
                                    continue;
                                }
                            };

                            for j in 0..m {
                                let pid = post_ids_ro[j];
                                if pid == 0 {
                                    buf[row_start + j] = 0;
                                } else if ds_types_ro[j] != Self::ACTIVE_ADS_DS_TYPE {
                                    buf[row_start + j] = 1;
                                } else {
                                    buf[row_start + j] =
                                        bloom_may_contain(bit_array, num_bits, pid) as u8;
                                }
                            }
                        }
                    });
                }
            });
        });

        Ok(())
    }

    pub fn reply(
        &mut self,
        py: Python<'_>,
        dataset_types: Vec<u32>,
        all_top_k_indices: Vec<Bound<'_, PyArray2<i32>>>,
        all_top_k_scores: Vec<Bound<'_, PyArray2<f32>>>,
        all_post_ids: Bound<'_, PyArray1<i64>>,
        all_author_ids: Bound<'_, PyArray1<i64>>,
        large_k: usize,
        batch_size: usize,
    ) -> PyResult<()> {
        let donated_indices: Vec<DonatedArray2<i32>> = all_top_k_indices
            .into_iter()
            .map(|a| DonatedArray2::new(a))
            .collect::<PyResult<Vec<_>>>()?;
        let donated_scores: Vec<DonatedArray2<f32>> = all_top_k_scores
            .into_iter()
            .map(|a| DonatedArray2::new(a))
            .collect::<PyResult<Vec<_>>>()?;
        let post_ids = DonatedArray1::new(all_post_ids)?;
        let author_ids = DonatedArray1::new(all_author_ids)?;

        let num_datasets = dataset_types.len();
        if donated_indices.len() != num_datasets || donated_scores.len() != num_datasets {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Mismatched array lengths: dataset_types={}, indices={}, scores={}",
                num_datasets,
                donated_indices.len(),
                donated_scores.len()
            )));
        }

        let items = std::mem::take(&mut self.items);

        py.detach(|| {
            thread::spawn(move || {
                let start = Instant::now();

                for (user_idx, mut item) in items.into_iter().enumerate() {
                    if user_idx >= batch_size {
                        if let Some(sender) = item.sender.take() {
                            sender
                                .send(Ok(pb::ScoredCandidates {
                                    user_id: item.user_id,
                                    candidates: vec![],
                                }))
                                .ok();
                        }
                        continue;
                    }

                    let user_top_k = item.top_k as usize;
                    let mut candidates: Vec<pb::ScoredCandidate> = Vec::new();

                    for (ds_idx, &ds_type) in dataset_types.iter().enumerate() {
                        let indices = &donated_indices[ds_idx];
                        let scores = &donated_scores[ds_idx];
                        let indices_view = unsafe { indices.as_array_view() };
                        let scores_view = unsafe { scores.as_array_view() };

                        let k = large_k.min(indices_view.ncols());
                        for j in 0..k {
                            let idx = indices_view[[user_idx, j]] as usize;
                            let pid = unsafe { post_ids.get(idx) };
                            if pid == 0 {
                                continue;
                            }
                            let aid = unsafe { author_ids.get(idx) };
                            let score = scores_view[[user_idx, j]];
                            candidates.push(pb::ScoredCandidate {
                                candidate: Some(pb::TweetInfo {
                                    tweet_id: pid as u64,
                                    author_id: aid as u64,
                                    ..Default::default()
                                }),
                                score,
                                source_idx: 0,
                                dataset_type: ds_type,
                            });
                        }
                    }

                    candidates.truncate(user_top_k);

                    let top_k_candidates = pb::ScoredCandidates {
                        user_id: item.user_id,
                        candidates,
                    };

                    if let Some(sender) = item.sender.take() {
                        sender.send(Ok(top_k_candidates)).ok();
                    }
                }

                let duration = start.elapsed();
                RETRIEVE_REPLY_LATENCY.observe(duration.as_secs_f64());
                log::debug!("RetrieveRequestBatch::reply took {:?}", duration);
            });
        });

        Ok(())
    }
}
fn serving_host_metadata() -> Option<(String, String)> {
    std::env::var_os("RETURN_SERVING_HOST_METADATA")?;
    Some((
        std::env::var("POD_NAME").unwrap_or_default(),
        std::env::var("NODE_NAME").unwrap_or_default(),
    ))
}

type ReloadDirective = (Vec<String>, String);

struct RecsysPredictorImpl {
    sender: mpsc::Sender<PredictRequestItem>,
    gauge: LoadGauge,
    inflight_semaphore: Semaphore,
    model_config: ModelConfig,
    reload_request: Arc<AtomicBool>,
    checkpoint_loading: Arc<AtomicBool>,
    current_checkpoint_timestamp: Arc<AtomicU64>,
    serving_prefix: Arc<StdMutex<String>>,
    peer_ready: Arc<AtomicBool>,
    peer_download_enabled: bool,
    reload_directive: Arc<StdMutex<Option<ReloadDirective>>>,
    enqueue_timeout_ms: u64,
    mm_embeddings_client: Option<MmEmbeddingsClient>,
    #[allow(dead_code)]
    sid_client: Option<Arc<crate::sid_client::SemanticIdClient>>,
    admission: Arc<AdmissionController>,
    #[allow(dead_code)]
    prefetch_mm_query_config: Option<Arc<PrefetchMmQueryConfig>>,
}

impl RecsysPredictorImpl {
    fn checkpoint_load_delay_seconds(&self) -> Option<u64> {
        std::env::var_os("RETURN_SERVING_HOST_METADATA")?;
        let created_secs = f64::from_bits(self.current_checkpoint_timestamp.load(Ordering::SeqCst));
        if created_secs <= 0.0 {
            return None;
        }
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        let delay_secs = now_secs - created_secs;
        if delay_secs < 0.0 {
            return None;
        }
        Some(delay_secs as u64)
    }

    async fn predict_next_actions_inner(
        &self,
        request: Request<pb::PredictNextActionsRequest>,
        start: Instant,
    ) -> tonic::Result<Response<pb::PredictNextActionsResponse>> {
        let request_id = extract_request_id(&request);
        let grpc_timeout = extract_grpc_timeout(&request);
        record_grpc_timeout_presence(grpc_timeout);
        let deadline = grpc_timeout.map(|t| start + t);

        let client = client_label(&request);
        record_client_server_network_delay(&request, &client);

        let cancel_token = CancellationToken::new();
        let cancel_guard: DropGuard = cancel_token.clone().drop_guard();
        let (complete_tx, complete_rx) = oneshot::channel::<()>();
        let mut request = request.into_inner();

        NUM_REQUESTS_RECEIVED.inc();
        record_not_ready_arrival(&client);

        if self.checkpoint_loading.load(Ordering::SeqCst) {
            log::warn!(
                "Request rejected during checkpoint loading for request_id: {}",
                request_id
            );
            NUM_REQUESTS_REJECTED
                .with_label_values(&["checkpoint_loading", client.as_str()])
                .inc();
            return Err(Status::unavailable(
                "Server is loading checkpoint, please retry on another pod",
            ));
        }

        let permit = self.inflight_semaphore.try_acquire().map_err(|_| {
            log::warn!(
                "Request rejected due to max concurrent requests limit for request_id: {}",
                request_id
            );
            NUM_REQUESTS_REJECTED
                .with_label_values(&["inflight_cap", client.as_str()])
                .inc();
            Status::resource_exhausted(
                "Maximum concurrent requests limit reached on recsys server.",
            )
        })?;

        let elapsed_us = start.elapsed().as_micros() as u64;
        if let Some(remaining_us) = remaining_deadline_budget_us(
            grpc_timeout,
            self.admission.fallback_budget_us(),
            elapsed_us,
        ) {
            ADMISSION_BUDGET_REMAINING
                .with_label_values(&[client.as_str()])
                .observe(remaining_us as f64 / 1000.0);
        }
        if let Some(reason) =
            should_shed_for_deadline(&self.admission, &self.sender, grpc_timeout, elapsed_us)
        {
            DEADLINE_SHED.with_label_values(&[client.as_str()]).inc();
            NUM_REQUESTS_REJECTED
                .with_label_values(&["deadline_admission", client.as_str()])
                .inc();
            log::warn!(
                "Request shed at admission for request_id: {}: {}",
                request_id,
                reason
            );
            return Err(Status::resource_exhausted(reason));
        }

        let m = request.candidate_sets.len();
        if m != 1 {
            log::warn!(
                "Request validation failed: expected 1 candidate set, got {} for request_id: {}",
                m,
                request_id
            );
            return Err(Status::invalid_argument(
                "Request must contain exactly one candidate set.",
            ));
        }

        let guard = self
            .gauge
            .try_acquire(request.load_ticket_id, 1, None)
            .ok_or_else(|| {
                log::warn!(
                    "Unable to acquire loadbalancer ticket for request_id: {}",
                    request_id
                );
                NUM_REQUESTS_REJECTED
                    .with_label_values(&["lb_ticket", client.as_str()])
                    .inc();
                Status::resource_exhausted("Unable to acquire loadbalancer ticket")
            })?;
        let _in_flight_guard = InFlightGuard::new();

        let sequence = request.sequences.pop();

        let columnar_sequence_bytes = request.columnar_sequences.pop().filter(|b| !b.is_empty());

        let candidate_set = match request.candidate_sets.pop() {
            Some(candidate_set) => candidate_set,
            _ => {
                return Err(Status::invalid_argument(
                    "Request does not contain any candidates.",
                ));
            }
        };
        let cancel_token_clone = cancel_token.clone();
        let request_id_clone = request_id.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = cancel_token_clone.cancelled() => {
                    NUM_REQUESTS_CANCELED.inc();
                    warn!("Request was cancelled for request_id: {}", request_id_clone);
                }
                _ = complete_rx => {
                }
            }
        });

        let distribution_sets = handle_request(
            candidate_set,
            sequence,
            columnar_sequence_bytes,
            request.return_logprob,
            request.top_logprobs_num,
            request.return_log_map,
            request.requested_action_indices,
            request.requested_continuous_action_indices,
            request.return_candidate_tweet_id_only,
            &self.sender,
            &self.model_config,
            &cancel_token,
            &request_id,
            self.enqueue_timeout_ms,
            &self.mm_embeddings_client,
            request.client_context,
            request.user_context,
            request.conversion_history,
            deadline,
        )
        .await;
        let response = match distribution_sets {
            Ok(ds) => {
                NUM_REQUESTS_SUCCEEDED.inc();
                REQUEST_LATENCY.observe(start.elapsed().as_secs_f64() * 1000.0);
                log::debug!(
                    "Request completed successfully for request_id: {} in {:?}",
                    request_id,
                    start.elapsed()
                );
                let (serving_pod_name, serving_node_name) = match serving_host_metadata() {
                    Some((pod, node)) => (Some(pod), Some(node)),
                    None => (None, None),
                };
                let checkpoint_load_delay_seconds = self.checkpoint_load_delay_seconds();
                Response::new(pb::PredictNextActionsResponse {
                    distribution_sets: vec![ds],
                    client_context: None,
                    serving_pod_name,
                    serving_node_name,
                    checkpoint_load_delay_seconds,
                })
            }
            Err(e) => {
                if e.code() == tonic::Code::Cancelled {
                    log::warn!(
                        "Request cancelled for request_id: {} with error: {:?}",
                        request_id,
                        e
                    );
                } else {
                    NUM_REQUESTS_FAILED.inc();
                    log::error!(
                        "Request failed for request_id: {} with error: {:?}",
                        request_id,
                        e
                    );
                }
                cancel_guard.disarm();
                return Err(e);
            }
        };
        drop(guard);
        drop(permit);
        cancel_guard.disarm();
        let _ = complete_tx.send(());

        Ok(response)
    }
}

#[tonic::async_trait]
impl pb::recsys_predictor_server::RecsysPredictor for RecsysPredictorImpl {
    async fn predict_next_actions(
        &self,
        request: Request<pb::PredictNextActionsRequest>,
    ) -> tonic::Result<Response<pb::PredictNextActionsResponse>> {
        let start = Instant::now();
        let client = client_label(&request);
        let workload = workload_label(&request);
        let src_dc = src_dc_label(&request);
        let result = self.predict_next_actions_inner(request, start).await;
        request_metrics::emit_request_metrics(
            start,
            status_code(&result),
            &client,
            &workload,
            &src_dc,
        );
        result
    }

    async fn reload_model(
        &self,
        request: Request<pb::ReloadModelRequest>,
    ) -> tonic::Result<Response<pb::ReloadModelResponse>> {
        let req = request.into_inner();
        if req.do_reload {
            log::info!("ReloadModel RPC called with do_reload=true - setting reload_request flag");
            if self.peer_download_enabled {
                *self.reload_directive.lock().unwrap() = if req.copy_urls.is_empty() {
                    None
                } else {
                    log::info!(
                        "ReloadModel P2P directive: {} peer url(s), target_prefix={:?}",
                        req.copy_urls.len(),
                        req.target_prefix
                    );
                    Some((req.copy_urls.clone(), req.target_prefix.clone()))
                };
            }
            self.reload_request.store(true, Ordering::SeqCst);
        } else {
            log::info!("ReloadModel RPC called with do_reload=false - poll mode, not reloading");
        }
        Ok(Response::new(pb::ReloadModelResponse {
            reload_in_progress: self.reload_request.load(Ordering::SeqCst),
            current_checkpoint_timestamp: f64::from_bits(
                self.current_checkpoint_timestamp.load(Ordering::SeqCst),
            ),
            serving_prefix: self.serving_prefix.lock().unwrap().clone(),
            peer_ready: self.peer_ready.load(Ordering::SeqCst),
        }))
    }
}

fn fill_semantic_ids(
    sids: &std::collections::HashMap<i64, Vec<i32>>,
    post_ids: &[i64],
    dst: &mut [u16],
    sid_dim: usize,
) {
    for (j, &post_id) in post_ids.iter().enumerate() {
        if post_id == 0 {
            continue;
        }
        if let Some(codes) = sids.get(&post_id) {
            assert_eq!(
                codes.len(),
                sid_dim,
                "SID codes length ({}) != sid_num_levels ({}) for post_id {}",
                codes.len(),
                sid_dim,
                post_id,
            );
            let dst_start = j * sid_dim;
            for (d, &c) in dst[dst_start..dst_start + sid_dim]
                .iter_mut()
                .zip(codes.iter())
            {
                *d = (c + 1) as u16;
            }
        }
    }
}

async fn handle_request(
    candidate_set: CandidateSet,
    sequence: Option<UserActionSequence>,
    columnar_sequence_bytes: Option<Bytes>,
    return_logprob: bool,
    top_logprobs_num: u32,
    return_log_map: bool,
    requested_action_indices: Vec<u32>,
    requested_continuous_action_indices: Vec<u32>,
    return_candidate_tweet_id_only: bool,
    sender: &mpsc::Sender<PredictRequestItem>,
    model_config: &ModelConfig,
    cancellation_token: &CancellationToken,
    request_id: &str,
    enqueue_timeout_ms: u64,
    mm_embeddings_client: &Option<MmEmbeddingsClient>,
    client_context: Option<pb::ClientContext>,
    user_context: Option<pb::UserContext>,
    conversion_history: Option<pb::ConversionHistoryContext>,
    deadline: Option<Instant>,
) -> tonic::Result<CandidateDistributionSet> {
    let (tx, rx) = oneshot::channel();

    let start = std::time::Instant::now();

    let input_buffer = if let Some(ref columnar_bytes) = columnar_sequence_bytes {
        log::debug!(
            "Computing input buffer from columnar bytes for request_id: {} ({} bytes)",
            request_id,
            columnar_bytes.len()
        );

        let mm_embeddings = if let Some(client) = mm_embeddings_client.as_ref() {
            let post_ids: Vec<u64> = candidate_set
                .candidates
                .iter()
                .map(|c| c.tweet_id)
                .collect();
            match client
                .fetch_mm_embeddings(
                    post_ids,
                    model_config.multimodal_embedding_dim,
                    model_config.candidate_seq_len,
                )
                .await
            {
                Ok(embeddings) => Some(embeddings),
                Err(e) => {
                    log::error!("Failed to prefetch MM embeddings: {}", e);
                    None
                }
            }
        } else {
            None
        };

        match InputBuffer::compute_from_columnar_bytes(
            model_config,
            columnar_bytes,
            &candidate_set,
            mm_embeddings,
            client_context.as_ref(),
            user_context.as_ref(),
            conversion_history.as_ref(),
        ) {
            Ok(buf) => buf,
            Err(e) => {
                log::error!(
                    "Failed to parse columnar bytes for request_id: {}, falling back to proto path: {}",
                    request_id,
                    e
                );
                InputBuffer::compute_for_item(
                    model_config,
                    &sequence,
                    &candidate_set,
                    None,
                    client_context.as_ref(),
                    user_context.as_ref(),
                    conversion_history.as_ref(),
                )
            }
        }
    } else {
        log::debug!("Computing input buffer for request_id: {}", request_id);

        let mm_embeddings = if let Some(client) = mm_embeddings_client.as_ref() {
            let post_ids: Vec<u64> = candidate_set
                .candidates
                .iter()
                .map(|c| c.tweet_id)
                .collect();
            match client
                .fetch_mm_embeddings(
                    post_ids,
                    model_config.multimodal_embedding_dim,
                    model_config.candidate_seq_len,
                )
                .await
            {
                Ok(embeddings) => Some(embeddings),
                Err(e) => {
                    log::error!("Failed to prefetch MM embeddings: {}", e);
                    None
                }
            }
        } else {
            None
        };

        InputBuffer::compute_for_item(
            model_config,
            &sequence,
            &candidate_set,
            mm_embeddings,
            client_context.as_ref(),
            user_context.as_ref(),
            conversion_history.as_ref(),
        )
    };
    let duration = start.elapsed();
    INPUT_BUFFER_COMPUTATION_TIME.observe(duration.as_secs_f64());
    log::debug!(
        "Input buffer computed in {:?} for request_id: {}",
        duration,
        request_id
    );

    let item = PredictRequestItem {
        sequence,
        return_logprob,
        top_logprobs_num,
        return_log_map,
        requested_action_indices,
        requested_continuous_action_indices,
        return_candidate_tweet_id_only,
        candidate_set,
        sender: Some(tx),
        input_buffer: Some(input_buffer),
        enqueue_time: Instant::now(),
        deadline,
        columnar_sequence_bytes,
    };

    tokio::select! {
        res = sender.send(item) => {
            res.map_err(|_| Status::internal("Failed to send item to worker"))?;
        }
        _ = tokio::time::sleep(Duration::from_millis(enqueue_timeout_ms)) => {
            BATCHING_TIMEOUT_DROP.inc();
            log::warn!("Request timed out while waiting to enqueue for request_id: {}", request_id);
            return Err(Status::resource_exhausted("Request timed out while waiting to enqueue (queue full)"));
        }
        _ = cancellation_token.cancelled() => {
            log::warn!("Request was cancelled during batching for request_id: {}", request_id);
            return Err(Status::cancelled("Request was cancelled"));
        }
    }

    tokio::select! {
        result = rx => match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(status)) => Err(status),
            Err(_recv_err) => {
                log::error!("Failed to receive result from worker for request_id: {}", request_id);
                Err(Status::internal("Failed to receive result from worker"))
            }
        },
        _ = cancellation_token.cancelled() =>  {
            log::warn!("Request was cancelled while waiting for worker result for request_id: {}", request_id);
            Err(Status::cancelled("Request was cancelled"))
        },
    }
}

pub trait PrepareBatch<T> {
    fn compute_batch_inputs_rust_quick<'py>(
        &self,
        py: Python<'py>,
        model_config: &ModelConfig,
        request: &T,
        _request_in_flight_id: usize,
        batch_size: usize,
    ) -> PyResult<bool>;
}

#[pyclass(module = "xai_recsys_engine")]
struct RankingBatchPrep {
    user_hashes: Py<PyArray2<i32>>,
    user_ip_hashes: Py<PyArray2<i32>>,
    history_post_hashes: Py<PyArray3<i32>>,
    history_auth_hashes: Py<PyArray3<i32>>,
    history_product_surfaces: Py<PyArray2<i32>>,
    history_actions: Py<PyArray3<bool>>,
    history_continuous_actions: Py<PyArray3<f32>>,
    candidate_post_hashes: Py<PyArray3<i32>>,
    candidate_auth_hashes: Py<PyArray3<i32>>,
    candidate_product_surfaces: Py<PyArray2<i32>>,
    candidate_embeddings: Option<Py<PyTuple>>,
    candidate_search_query_embeddings: Option<Py<PyArray3<f32>>>,
    user_categorical_features: Option<Py<PyArray2<i16>>>,
    user_bool_features: Option<Py<PyArray2<bool>>>,
    user_float_features: Option<Py<PyArray2<f32>>>,
    user_int64_features: Option<Py<PyArray2<i64>>>,
    user_installed_apps: Option<Py<PyArray2<bool>>>,
    history_categorical_features: Option<Py<PyArray3<i16>>>,
    history_bool_features: Option<Py<PyArray3<bool>>>,
    history_float_features: Option<Py<PyArray3<f32>>>,
    history_int64_features: Option<Py<PyArray3<i64>>>,
    candidate_categorical_features: Option<Py<PyArray3<i16>>>,
    candidate_bool_features: Option<Py<PyArray3<bool>>>,
    candidate_float_features: Option<Py<PyArray3<f32>>>,
    candidate_int64_features: Option<Py<PyArray3<i64>>>,
    history_impr_ts: Option<Py<PyArray2<i32>>>,
    history_post_creation_ts_sec: Option<Py<PyArray2<i32>>>,
    candidate_impr_ts: Option<Py<PyArray2<i32>>>,
    candidate_post_creation_ts_sec: Option<Py<PyArray2<i32>>>,
    candidate_line_item_ids: Option<Py<PyArray2<i64>>>,
    candidate_campaign_ids: Option<Py<PyArray2<i64>>>,
    candidate_funding_instrument_ids: Option<Py<PyArray2<i64>>>,
    user_client_id_idx: Option<Py<PyArray1<i32>>>,
    user_country_code_idx: Option<Py<PyArray1<i32>>>,
    user_gender_idx: Option<Py<PyArray1<i32>>>,
    user_age_bucket_idx: Option<Py<PyArray1<i32>>>,
    user_age_in_years_idx: Option<Py<PyArray1<i32>>>,
    user_inferred_gender_idx: Option<Py<PyArray1<i32>>>,
    user_inferred_gender_score: Option<Py<PyArray1<f32>>>,
    candidate_conversion_dense_features: Option<Py<PyArray3<f32>>>,
    user_conversion_history_hashes: Option<Py<PyArray2<i32>>>,
    candidate_account_hashes: Option<Py<PyArray2<i32>>>,
    history_post_sids: Option<Py<PyArray3<u16>>>,
    candidate_post_sids: Option<Py<PyArray3<u16>>>,
}

#[pymethods]
impl RankingBatchPrep {
    #[new]
    #[pyo3(signature = (
        user_hashes,
        user_ip_hashes,
        history_post_hashes,
        history_auth_hashes,
        history_product_surfaces,
        history_actions,
        history_continuous_actions,
        candidate_post_hashes,
        candidate_auth_hashes,
        candidate_product_surfaces,
        candidate_embeddings,
        candidate_search_query_embeddings = None,
        user_categorical_features = None,
        user_bool_features = None,
        user_float_features = None,
        user_int64_features = None,
        user_installed_apps = None,
        history_categorical_features = None,
        history_bool_features = None,
        history_float_features = None,
        history_int64_features = None,
        candidate_categorical_features = None,
        candidate_bool_features = None,
        candidate_float_features = None,
        candidate_int64_features = None,
        history_impr_ts = None,
        history_post_creation_ts_sec = None,
        candidate_impr_ts = None,
        candidate_post_creation_ts_sec = None,
        candidate_line_item_ids = None,
        candidate_campaign_ids = None,
        candidate_funding_instrument_ids = None,
        user_client_id_idx = None,
        user_country_code_idx = None,
        user_gender_idx = None,
        user_age_bucket_idx = None,
        user_age_in_years_idx = None,
        user_inferred_gender_idx = None,
        user_inferred_gender_score = None,
        candidate_conversion_dense_features = None,
        user_conversion_history_hashes = None,
        candidate_account_hashes = None,
        history_post_sids = None,
        candidate_post_sids = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        user_hashes: Py<PyArray2<i32>>,
        user_ip_hashes: Py<PyArray2<i32>>,
        history_post_hashes: Py<PyArray3<i32>>,
        history_auth_hashes: Py<PyArray3<i32>>,
        history_product_surfaces: Py<PyArray2<i32>>,
        history_actions: Py<PyArray3<bool>>,
        history_continuous_actions: Py<PyArray3<f32>>,
        candidate_post_hashes: Py<PyArray3<i32>>,
        candidate_auth_hashes: Py<PyArray3<i32>>,
        candidate_product_surfaces: Py<PyArray2<i32>>,
        candidate_embeddings: Option<Py<PyTuple>>,
        candidate_search_query_embeddings: Option<Py<PyArray3<f32>>>,
        user_categorical_features: Option<Py<PyArray2<i16>>>,
        user_bool_features: Option<Py<PyArray2<bool>>>,
        user_float_features: Option<Py<PyArray2<f32>>>,
        user_int64_features: Option<Py<PyArray2<i64>>>,
        user_installed_apps: Option<Py<PyArray2<bool>>>,
        history_categorical_features: Option<Py<PyArray3<i16>>>,
        history_bool_features: Option<Py<PyArray3<bool>>>,
        history_float_features: Option<Py<PyArray3<f32>>>,
        history_int64_features: Option<Py<PyArray3<i64>>>,
        candidate_categorical_features: Option<Py<PyArray3<i16>>>,
        candidate_bool_features: Option<Py<PyArray3<bool>>>,
        candidate_float_features: Option<Py<PyArray3<f32>>>,
        candidate_int64_features: Option<Py<PyArray3<i64>>>,
        history_impr_ts: Option<Py<PyArray2<i32>>>,
        history_post_creation_ts_sec: Option<Py<PyArray2<i32>>>,
        candidate_impr_ts: Option<Py<PyArray2<i32>>>,
        candidate_post_creation_ts_sec: Option<Py<PyArray2<i32>>>,
        candidate_line_item_ids: Option<Py<PyArray2<i64>>>,
        candidate_campaign_ids: Option<Py<PyArray2<i64>>>,
        candidate_funding_instrument_ids: Option<Py<PyArray2<i64>>>,
        user_client_id_idx: Option<Py<PyArray1<i32>>>,
        user_country_code_idx: Option<Py<PyArray1<i32>>>,
        user_gender_idx: Option<Py<PyArray1<i32>>>,
        user_age_bucket_idx: Option<Py<PyArray1<i32>>>,
        user_age_in_years_idx: Option<Py<PyArray1<i32>>>,
        user_inferred_gender_idx: Option<Py<PyArray1<i32>>>,
        user_inferred_gender_score: Option<Py<PyArray1<f32>>>,
        candidate_conversion_dense_features: Option<Py<PyArray3<f32>>>,
        user_conversion_history_hashes: Option<Py<PyArray2<i32>>>,
        candidate_account_hashes: Option<Py<PyArray2<i32>>>,
        history_post_sids: Option<Py<PyArray3<u16>>>,
        candidate_post_sids: Option<Py<PyArray3<u16>>>,
    ) -> Self {
        RankingBatchPrep {
            user_hashes,
            user_ip_hashes,
            history_post_hashes,
            history_auth_hashes,
            history_product_surfaces,
            history_actions,
            history_continuous_actions,
            candidate_post_hashes,
            candidate_auth_hashes,
            candidate_product_surfaces,
            candidate_embeddings,
            candidate_search_query_embeddings,
            user_categorical_features,
            user_bool_features,
            user_float_features,
            user_int64_features,
            user_installed_apps,
            history_categorical_features,
            history_bool_features,
            history_float_features,
            history_int64_features,
            candidate_categorical_features,
            candidate_bool_features,
            candidate_float_features,
            candidate_int64_features,
            history_impr_ts,
            history_post_creation_ts_sec,
            candidate_impr_ts,
            candidate_post_creation_ts_sec,
            candidate_line_item_ids,
            candidate_campaign_ids,
            candidate_funding_instrument_ids,
            user_client_id_idx,
            user_country_code_idx,
            user_gender_idx,
            user_age_bucket_idx,
            user_age_in_years_idx,
            user_inferred_gender_idx,
            user_inferred_gender_score,
            candidate_conversion_dense_features,
            user_conversion_history_hashes,
            candidate_account_hashes,
            history_post_sids,
            candidate_post_sids,
        }
    }
}

impl PrepareBatch<PredictRequestBatch> for RankingBatchPrep {
    fn compute_batch_inputs_rust_quick<'py>(
        &self,
        py: Python<'py>,
        model_config: &ModelConfig,
        request: &PredictRequestBatch,
        _request_in_flight_id: usize,
        batch_size: usize,
    ) -> PyResult<bool> {
        let length_of_input = request.items.len().min(batch_size);

        if length_of_input == 0 {
            return Ok(false);
        }

        let mut user_hashes = self.user_hashes.bind(py).readwrite();
        let mut user_ip_hashes = self.user_ip_hashes.bind(py).readwrite();
        let mut history_post_hashes = self.history_post_hashes.bind(py).readwrite();
        let mut history_auth_hashes = self.history_auth_hashes.bind(py).readwrite();
        let mut history_product_surfaces = self.history_product_surfaces.bind(py).readwrite();
        let mut history_actions = self.history_actions.bind(py).readwrite();
        let mut history_continuous_actions = self.history_continuous_actions.bind(py).readwrite();
        let mut candidate_post_hashes = self.candidate_post_hashes.bind(py).readwrite();
        let mut candidate_auth_hashes = self.candidate_auth_hashes.bind(py).readwrite();
        let mut candidate_product_surfaces = self.candidate_product_surfaces.bind(py).readwrite();
        let candidate_embeddings_tuple = self.candidate_embeddings.as_ref().map(|t| t.bind(py));
        let mut candidate_embeddings_arrays = Vec::new();
        if let Some(tuple) = &candidate_embeddings_tuple {
            for item in tuple.iter() {
                let arr: &Bound<'_, PyArray1<f16>> = item.downcast()?;
                candidate_embeddings_arrays.push(arr.readwrite());
            }
        }
        let mut candidate_search_query_embeddings_arr = self
            .candidate_search_query_embeddings
            .as_ref()
            .map(|sq| sq.bind(py).readwrite());

        let user_hashes_slice = user_hashes.as_slice_mut()?;
        let user_ip_hashes_slice = user_ip_hashes.as_slice_mut()?;
        let history_post_hashes_slice = history_post_hashes.as_slice_mut()?;
        let history_auth_hashes_slice = history_auth_hashes.as_slice_mut()?;
        let history_product_surfaces_slice = history_product_surfaces.as_slice_mut()?;
        let history_actions_slice = history_actions.as_slice_mut()?;
        let history_continuous_actions_slice = history_continuous_actions.as_slice_mut()?;
        let candidate_post_hashes_slice = candidate_post_hashes.as_slice_mut()?;
        let candidate_auth_hashes_slice = candidate_auth_hashes.as_slice_mut()?;
        let candidate_product_surfaces_slice = candidate_product_surfaces.as_slice_mut()?;
        let mut hist_sid_rw = self
            .history_post_sids
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_sid_slice = hist_sid_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut cand_sid_rw = self
            .candidate_post_sids
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_sid_slice = cand_sid_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut candidate_embeddings_slices: Vec<&mut [f16]> = Vec::new();
        for arr in candidate_embeddings_arrays.iter_mut() {
            candidate_embeddings_slices.push(arr.as_slice_mut()?);
        }
        let candidate_search_query_embeddings_slice = candidate_search_query_embeddings_arr
            .as_mut()
            .map(|arr| arr.as_slice_mut())
            .transpose()?;
        let mut user_categorical_features_arr = self
            .user_categorical_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_bool_features_arr = self
            .user_bool_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_float_features_arr = self
            .user_float_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_int64_features_arr = self
            .user_int64_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_installed_apps_arr = self
            .user_installed_apps
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let user_categorical_slice = user_categorical_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_bool_slice = user_bool_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_float_slice = user_float_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_int64_slice = user_int64_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_installed_apps_slice = user_installed_apps_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let mut hist_cat_arr = self
            .history_categorical_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_bool_arr = self
            .history_bool_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_float_arr = self
            .history_float_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_int64_arr = self
            .history_int64_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_cat_arr = self
            .candidate_categorical_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_bool_arr = self
            .candidate_bool_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_float_arr = self
            .candidate_float_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_int64_arr = self
            .candidate_int64_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let hist_cat_slice = hist_cat_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let hist_bool_slice = hist_bool_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let hist_float_slice = hist_float_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let hist_int64_slice = hist_int64_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let cand_cat_slice = cand_cat_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let cand_bool_slice = cand_bool_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let cand_float_slice = cand_float_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let cand_int64_slice = cand_int64_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;

        let mut hist_impr_ts_rw = self
            .history_impr_ts
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_post_creation_ts_sec_rw = self
            .history_post_creation_ts_sec
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_impr_ts_rw = self
            .candidate_impr_ts
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_post_creation_ts_sec_rw = self
            .candidate_post_creation_ts_sec
            .as_ref()
            .map(|a| a.bind(py).readwrite());

        let mut hist_impr_ts_slice = hist_impr_ts_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut hist_post_creation_ts_sec_slice = hist_post_creation_ts_sec_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut cand_impr_ts_slice = cand_impr_ts_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut cand_post_creation_ts_sec_slice = cand_post_creation_ts_sec_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;

        let mut cand_line_item_ids_rw = self
            .candidate_line_item_ids
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_campaign_ids_rw = self
            .candidate_campaign_ids
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_funding_instrument_ids_rw = self
            .candidate_funding_instrument_ids
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_line_item_ids_slice = cand_line_item_ids_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut cand_campaign_ids_slice = cand_campaign_ids_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut cand_funding_instrument_ids_slice = cand_funding_instrument_ids_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;

        let mut user_client_id_idx_rw = self
            .user_client_id_idx
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_country_code_idx_rw = self
            .user_country_code_idx
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_gender_idx_rw = self
            .user_gender_idx
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_age_bucket_idx_rw = self
            .user_age_bucket_idx
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_age_in_years_idx_rw = self
            .user_age_in_years_idx
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_inferred_gender_idx_rw = self
            .user_inferred_gender_idx
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_inferred_gender_score_rw = self
            .user_inferred_gender_score
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_client_id_idx_slice = user_client_id_idx_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_country_code_idx_slice = user_country_code_idx_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_gender_idx_slice = user_gender_idx_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_age_bucket_idx_slice = user_age_bucket_idx_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_age_in_years_idx_slice = user_age_in_years_idx_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_inferred_gender_idx_slice = user_inferred_gender_idx_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_inferred_gender_score_slice = user_inferred_gender_score_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;

        let mut candidate_conversion_dense_rw = self
            .candidate_conversion_dense_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_conversion_hashes_rw = self
            .user_conversion_history_hashes
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut cand_account_hashes_rw = self
            .candidate_account_hashes
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut candidate_conversion_dense_slice = candidate_conversion_dense_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut user_conversion_hashes_slice = user_conversion_hashes_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;
        let mut cand_account_hashes_slice = cand_account_hashes_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;

        let num_user_hashes = model_config.hash_table.num_user_hashes();
        let num_ip_hashes = model_config.hash_table.num_ip_hashes();
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let candidate_seq_len = model_config.candidate_seq_len;
        let history_seq_len = model_config.history_seq_len;
        let output_vocab_size = model_config.hash_table.output_vocab_size;
        let num_continuous_actions = model_config.hash_table.num_continuous_actions;
        let embedding_dim = model_config.multimodal_embedding_dim;
        let search_query_embedding_dim = model_config.hash_table.search_query_embedding_dim;
        let sid_num_levels = model_config.sid_num_levels;

        py.detach(|| {
            use rayon::prelude::*;

            user_hashes_slice
                .par_chunks_exact_mut(num_user_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(user_hash_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        user_hash_row.copy_from_slice(&input_buffer.user_hashes);
                    }
                });

            if num_ip_hashes > 0 {
                user_ip_hashes_slice
                    .par_chunks_exact_mut(num_ip_hashes)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(ip_hash_row, item)| {
                        if let Some(ref input_buffer) = item.input_buffer {
                            ip_hash_row.copy_from_slice(&input_buffer.user_ip_hashes);
                        }
                    });
            }

            history_post_hashes_slice
                .par_chunks_exact_mut(history_seq_len * num_item_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(post_hash_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        post_hash_row.copy_from_slice(&input_buffer.history_post_hashes);
                    }
                });

            history_auth_hashes_slice
                .par_chunks_exact_mut(history_seq_len * num_author_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(auth_hash_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        auth_hash_row.copy_from_slice(&input_buffer.history_auth_hashes);
                    }
                });

            history_product_surfaces_slice
                .par_chunks_exact_mut(history_seq_len)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(product_surface_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        product_surface_row.copy_from_slice(&input_buffer.history_product_surfaces);
                    }
                });

            history_actions_slice
                .par_chunks_exact_mut(history_seq_len * output_vocab_size)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(actions_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        actions_row.copy_from_slice(&input_buffer.history_actions);
                    }
                });

            history_continuous_actions_slice
                .par_chunks_exact_mut(history_seq_len * num_continuous_actions)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(continuous_actions_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        continuous_actions_row
                            .copy_from_slice(&input_buffer.history_continuous_actions);
                    }
                });

            candidate_post_hashes_slice
                .par_chunks_exact_mut(candidate_seq_len * num_item_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(candidate_post_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        candidate_post_row.copy_from_slice(&input_buffer.candidate_post_hashes);
                    }
                });

            candidate_auth_hashes_slice
                .par_chunks_exact_mut(candidate_seq_len * num_author_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(candidate_auth_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        candidate_auth_row.copy_from_slice(&input_buffer.candidate_auth_hashes);
                    }
                });

            candidate_product_surfaces_slice
                .par_chunks_exact_mut(candidate_seq_len)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(product_surface_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        product_surface_row
                            .copy_from_slice(&input_buffer.candidate_product_surfaces);
                    }
                });

            if sid_num_levels > 0 {
                if let Some(ref mut sl) = hist_sid_slice {
                    sl.par_chunks_exact_mut(history_seq_len * sid_num_levels)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.history_semantic_ids);
                            }
                        });
                }
                if let Some(ref mut sl) = cand_sid_slice {
                    sl.par_chunks_exact_mut(candidate_seq_len * sid_num_levels)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.candidate_semantic_ids);
                            }
                        });
                }
            }

            if embedding_dim > 0 && !candidate_embeddings_slices.is_empty() {
                let row_size = candidate_seq_len * embedding_dim;
                let num_shards = candidate_embeddings_slices.len();
                assert!(
                    batch_size.is_multiple_of(num_shards),
                    "batch_size ({batch_size}) must be evenly divisible by num_shards ({num_shards})"
                );
                let chunk_size = batch_size / num_shards;

                request
                    .items
                    .par_chunks(chunk_size)
                    .zip(candidate_embeddings_slices.par_iter_mut())
                    .for_each(|(items_chunk, shard_slice)| {
                        shard_slice
                            .par_chunks_exact_mut(row_size)
                            .zip(items_chunk.par_iter())
                            .for_each(|(embedding_row, item)| {
                                if let Some(ref input_buffer) = item.input_buffer {
                                    let src = &input_buffer.candidate_embeddings;
                                    let copy_len = embedding_row.len().min(src.len());
                                    embedding_row[..copy_len].copy_from_slice(&src[..copy_len]);
                                }
                            });
                    });
            }

            if search_query_embedding_dim > 0
                && let Some(sq_slice) = candidate_search_query_embeddings_slice
            {
                sq_slice
                    .par_chunks_exact_mut(candidate_seq_len * search_query_embedding_dim)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(search_query_emb_row, item)| {
                        if let Some(ref input_buffer) = item.input_buffer {
                            search_query_emb_row
                                .copy_from_slice(&input_buffer.candidate_search_query_embeddings);
                        }
                    });
            }

            let n_user_cat = model_config.hash_table.num_user_categorical_features;
            if n_user_cat > 0
                && let Some(sl) = user_categorical_slice {
                    sl.par_chunks_exact_mut(n_user_cat)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.user_categorical_features);
                            }
                        });
                }
            let n_user_bool = model_config.hash_table.num_user_bool_features;
            if n_user_bool > 0
                && let Some(sl) = user_bool_slice {
                    sl.par_chunks_exact_mut(n_user_bool)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.user_bool_features);
                            }
                        });
                }
            let n_user_float = model_config.hash_table.num_user_float_features;
            if n_user_float > 0
                && let Some(sl) = user_float_slice {
                    sl.par_chunks_exact_mut(n_user_float)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.user_float_features);
                            }
                        });
                }
            let n_user_int64 = model_config.hash_table.num_user_int64_features;
            if n_user_int64 > 0
                && let Some(sl) = user_int64_slice {
                    sl.par_chunks_exact_mut(n_user_int64)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.user_int64_features);
                            }
                        });
                }

            let n_user_installed_apps = model_config.hash_table.num_user_installed_apps;
            if n_user_installed_apps > 0
                && let Some(sl) = user_installed_apps_slice {
                    sl.par_chunks_exact_mut(n_user_installed_apps)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.user_installed_apps);
                            }
                        });
                }

            let n_post_cat = model_config.hash_table.num_post_categorical_features;
            if n_post_cat > 0
                && let Some(sl) = hist_cat_slice {
                    sl.par_chunks_exact_mut(history_seq_len * n_post_cat)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.history_categorical_features);
                            }
                        });
                }
            let n_post_bool = model_config.hash_table.num_post_bool_features;
            if n_post_bool > 0
                && let Some(sl) = hist_bool_slice {
                    sl.par_chunks_exact_mut(history_seq_len * n_post_bool)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.history_bool_features);
                            }
                        });
                }
            let n_post_float = model_config.hash_table.num_post_float_features;
            if n_post_float > 0
                && let Some(sl) = hist_float_slice {
                    sl.par_chunks_exact_mut(history_seq_len * n_post_float)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.history_float_features);
                            }
                        });
                }
            let n_post_int64 = model_config.hash_table.num_post_int64_features;
            if n_post_int64 > 0
                && let Some(sl) = hist_int64_slice {
                    sl.par_chunks_exact_mut(history_seq_len * n_post_int64)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.history_int64_features);
                            }
                        });
                }

            if n_post_cat > 0
                && let Some(sl) = cand_cat_slice {
                    sl.par_chunks_exact_mut(candidate_seq_len * n_post_cat)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.candidate_categorical_features);
                            }
                        });
                }
            if n_post_bool > 0
                && let Some(sl) = cand_bool_slice {
                    sl.par_chunks_exact_mut(candidate_seq_len * n_post_bool)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.candidate_bool_features);
                            }
                        });
                }
            if n_post_float > 0
                && let Some(sl) = cand_float_slice {
                    sl.par_chunks_exact_mut(candidate_seq_len * n_post_float)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.candidate_float_features);
                            }
                        });
                }
            if n_post_int64 > 0
                && let Some(sl) = cand_int64_slice {
                    sl.par_chunks_exact_mut(candidate_seq_len * n_post_int64)
                        .take(length_of_input)
                        .zip(request.items.par_iter())
                        .for_each(|(row, item)| {
                            if let Some(ref buf) = item.input_buffer {
                                row.copy_from_slice(&buf.candidate_int64_features);
                            }
                        });
                }

            macro_rules! copy_perceived_i32 {
                ($opt_slice:expr, $seq_len:expr, $field:ident) => {
                    if let Some(ref mut sl) = $opt_slice {
                        sl.par_chunks_exact_mut($seq_len)
                            .take(length_of_input)
                            .zip(request.items.par_iter())
                            .for_each(|(row, item)| {
                                if let Some(ref ib) = item.input_buffer {
                                    row.copy_from_slice(&ib.$field);
                                }
                            });
                    }
                };
            }

            copy_perceived_i32!(hist_impr_ts_slice, history_seq_len, history_impr_ts);
            copy_perceived_i32!(hist_post_creation_ts_sec_slice, history_seq_len, history_post_creation_ts_sec);
            copy_perceived_i32!(cand_impr_ts_slice, candidate_seq_len, candidate_impr_ts);
            copy_perceived_i32!(cand_post_creation_ts_sec_slice, candidate_seq_len, candidate_post_creation_ts_sec);
            copy_perceived_i32!(cand_line_item_ids_slice, candidate_seq_len, candidate_line_item_ids);
            copy_perceived_i32!(cand_campaign_ids_slice, candidate_seq_len, candidate_campaign_ids);
            copy_perceived_i32!(cand_funding_instrument_ids_slice, candidate_seq_len, candidate_funding_instrument_ids);
            copy_perceived_i32!(user_client_id_idx_slice, 1, user_client_id_idx);
            copy_perceived_i32!(user_country_code_idx_slice, 1, user_country_code_idx);
            copy_perceived_i32!(user_gender_idx_slice, 1, user_gender_idx);
            copy_perceived_i32!(user_age_bucket_idx_slice, 1, user_age_bucket_idx);
            copy_perceived_i32!(user_age_in_years_idx_slice, 1, user_age_in_years_idx);
            copy_perceived_i32!(user_inferred_gender_idx_slice, 1, user_inferred_gender_idx);
            copy_perceived_i32!(user_inferred_gender_score_slice, 1, user_inferred_gender_score);

            copy_perceived_i32!(candidate_conversion_dense_slice, candidate_seq_len * NUM_CONVERSION_DENSE_FEATURES, candidate_conversion_dense_features);
            copy_perceived_i32!(user_conversion_hashes_slice, NUM_CONVERSION_HISTORY_HASHES, user_conversion_history_hashes);
            copy_perceived_i32!(cand_account_hashes_slice, candidate_seq_len, candidate_account_hashes);
        });

        let history_hist = TOKENS_PER_ROW.with_label_values(&["history"]);
        let candidate_hist = TOKENS_PER_ROW.with_label_values(&["candidate"]);
        let mut history_tokens: usize = 0;
        let mut candidate_tokens: usize = 0;
        for item in &request.items[..length_of_input] {
            let num_history = item.input_buffer.as_ref().map_or(0, |b| b.num_history);
            let num_candidates = item.candidate_set.candidates.len().min(candidate_seq_len);
            history_hist.observe(num_history as f64);
            candidate_hist.observe(num_candidates as f64);
            history_tokens += num_history;
            candidate_tokens += num_candidates;
        }
        SEQUENCES_PROCESSED.inc_by(length_of_input as u64);
        TOKENS_PROCESSED
            .with_label_values(&["history"])
            .inc_by(history_tokens as u64);
        TOKENS_PROCESSED
            .with_label_values(&["candidate"])
            .inc_by(candidate_tokens as u64);
        TOKENS_PADDED.inc_by((length_of_input * (history_seq_len + candidate_seq_len)) as u64);

        Ok(true)
    }
}

#[pyclass(module = "xai_recsys_engine", get_all)]
struct RetrievalBatchPrep {
    user_hashes: Py<PyArray2<i32>>,
    user_ip_hashes: Py<PyArray2<i32>>,
    history_post_hashes: Py<PyArray3<i32>>,
    history_auth_hashes: Py<PyArray3<i32>>,
    history_actions: Py<PyArray3<bool>>,
    history_continuous_actions: Py<PyArray3<f32>>,
    history_product_surfaces: Py<PyArray2<i32>>,
    user_categorical_features: Option<Py<PyArray2<i16>>>,
    user_bool_features: Option<Py<PyArray2<bool>>>,
    user_float_features: Option<Py<PyArray2<f32>>>,
    user_int64_features: Option<Py<PyArray2<i64>>>,
    user_installed_apps: Option<Py<PyArray2<bool>>>,
    history_categorical_features: Option<Py<PyArray3<i16>>>,
    history_bool_features: Option<Py<PyArray3<bool>>>,
    history_float_features: Option<Py<PyArray3<f32>>>,
    history_int64_features: Option<Py<PyArray3<i64>>>,
    history_post_ids: Py<PyArray2<i64>>,
    history_post_sids: Option<Py<PyArray3<u16>>>,
}

#[pymethods]
impl RetrievalBatchPrep {
    #[new]
    #[pyo3(signature = (
        user_hashes,
        user_ip_hashes,
        history_post_hashes,
        history_auth_hashes,
        history_actions,
        history_continuous_actions,
        history_product_surfaces,
        history_post_ids,
        user_categorical_features = None,
        user_bool_features = None,
        user_float_features = None,
        user_int64_features = None,
        user_installed_apps = None,
        history_categorical_features = None,
        history_bool_features = None,
        history_float_features = None,
        history_int64_features = None,
        history_post_sids = None,
    ))]
    fn new(
        user_hashes: Py<PyArray2<i32>>,
        user_ip_hashes: Py<PyArray2<i32>>,
        history_post_hashes: Py<PyArray3<i32>>,
        history_auth_hashes: Py<PyArray3<i32>>,
        history_actions: Py<PyArray3<bool>>,
        history_continuous_actions: Py<PyArray3<f32>>,
        history_product_surfaces: Py<PyArray2<i32>>,
        history_post_ids: Py<PyArray2<i64>>,
        user_categorical_features: Option<Py<PyArray2<i16>>>,
        user_bool_features: Option<Py<PyArray2<bool>>>,
        user_float_features: Option<Py<PyArray2<f32>>>,
        user_int64_features: Option<Py<PyArray2<i64>>>,
        user_installed_apps: Option<Py<PyArray2<bool>>>,
        history_categorical_features: Option<Py<PyArray3<i16>>>,
        history_bool_features: Option<Py<PyArray3<bool>>>,
        history_float_features: Option<Py<PyArray3<f32>>>,
        history_int64_features: Option<Py<PyArray3<i64>>>,
        history_post_sids: Option<Py<PyArray3<u16>>>,
    ) -> Self {
        RetrievalBatchPrep {
            user_hashes,
            user_ip_hashes,
            history_post_hashes,
            history_auth_hashes,
            history_actions,
            history_continuous_actions,
            history_product_surfaces,
            user_categorical_features,
            user_bool_features,
            user_float_features,
            user_int64_features,
            user_installed_apps,
            history_categorical_features,
            history_bool_features,
            history_float_features,
            history_int64_features,
            history_post_ids,
            history_post_sids,
        }
    }
}

impl PrepareBatch<RetrieveRequestBatch> for RetrievalBatchPrep {
    fn compute_batch_inputs_rust_quick<'py>(
        &self,
        py: Python<'py>,
        model_config: &ModelConfig,
        request: &RetrieveRequestBatch,
        _request_in_flight_id: usize,
        batch_size: usize,
    ) -> PyResult<bool> {
        let length_of_input = request.items.len().min(batch_size);

        if length_of_input == 0 {
            return Ok(false);
        }

        let mut user_hashes = self.user_hashes.bind(py).readwrite();
        let mut user_ip_hashes = self.user_ip_hashes.bind(py).readwrite();
        let mut history_post_hashes = self.history_post_hashes.bind(py).readwrite();
        let mut history_auth_hashes = self.history_auth_hashes.bind(py).readwrite();
        let mut history_actions = self.history_actions.bind(py).readwrite();
        let mut history_continuous_actions = self.history_continuous_actions.bind(py).readwrite();
        let mut history_product_surfaces = self.history_product_surfaces.bind(py).readwrite();
        let mut history_post_ids = self.history_post_ids.bind(py).readwrite();

        let mut hist_sid_rw = self
            .history_post_sids
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_sid_slice = hist_sid_rw
            .as_mut()
            .map(|rw| rw.as_slice_mut())
            .transpose()?;

        let user_hashes_slice = user_hashes.as_slice_mut()?;
        let user_ip_hashes_slice = user_ip_hashes.as_slice_mut()?;
        let history_post_hashes_slice = history_post_hashes.as_slice_mut()?;
        let history_auth_hashes_slice = history_auth_hashes.as_slice_mut()?;
        let history_actions_slice = history_actions.as_slice_mut()?;
        let history_continuous_actions_slice = history_continuous_actions.as_slice_mut()?;
        let history_product_surfaces_slice = history_product_surfaces.as_slice_mut()?;
        let history_post_ids_slice = history_post_ids.as_slice_mut()?;
        let mut user_categorical_features_arr = self
            .user_categorical_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_bool_features_arr = self
            .user_bool_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_float_features_arr = self
            .user_float_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_int64_features_arr = self
            .user_int64_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut user_installed_apps_arr = self
            .user_installed_apps
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let user_categorical_slice = user_categorical_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_bool_slice = user_bool_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_float_slice = user_float_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_int64_slice = user_int64_features_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let user_installed_apps_slice = user_installed_apps_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let mut hist_cat_arr = self
            .history_categorical_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_bool_arr = self
            .history_bool_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_float_arr = self
            .history_float_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let mut hist_int64_arr = self
            .history_int64_features
            .as_ref()
            .map(|a| a.bind(py).readwrite());
        let hist_cat_slice = hist_cat_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let hist_bool_slice = hist_bool_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let hist_float_slice = hist_float_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;
        let hist_int64_slice = hist_int64_arr
            .as_mut()
            .map(|a| a.as_slice_mut())
            .transpose()?;

        let num_user_hashes = model_config.hash_table.num_user_hashes();
        let num_ip_hashes = model_config.hash_table.num_ip_hashes();
        let num_author_hashes = model_config.hash_table.num_author_hashes();
        let num_item_hashes = model_config.hash_table.num_item_hashes();
        let history_seq_len = model_config.history_seq_len;
        let output_vocab_size = model_config.hash_table.output_vocab_size;
        let num_continuous_actions = model_config.hash_table.num_continuous_actions;

        py.detach(|| {
            use rayon::prelude::*;

            user_hashes_slice
                .par_chunks_exact_mut(num_user_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(user_hash_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        user_hash_row.copy_from_slice(&input_buffer.user_hashes);
                    }
                });

            if num_ip_hashes > 0 {
                user_ip_hashes_slice
                    .par_chunks_exact_mut(num_ip_hashes)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(ip_hash_row, item)| {
                        if let Some(ref input_buffer) = item.input_buffer {
                            ip_hash_row.copy_from_slice(&input_buffer.user_ip_hashes);
                        }
                    });
            }

            history_post_hashes_slice
                .par_chunks_exact_mut(history_seq_len * num_item_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(post_hash_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        post_hash_row.copy_from_slice(&input_buffer.history_post_hashes);
                    }
                });

            history_auth_hashes_slice
                .par_chunks_exact_mut(history_seq_len * num_author_hashes)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(auth_hash_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        auth_hash_row.copy_from_slice(&input_buffer.history_auth_hashes);
                    }
                });

            history_actions_slice
                .par_chunks_exact_mut(history_seq_len * output_vocab_size)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(actions_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        actions_row.copy_from_slice(&input_buffer.history_actions);
                    }
                });

            if num_continuous_actions > 0 {
                history_continuous_actions_slice
                    .par_chunks_exact_mut(history_seq_len * num_continuous_actions)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(continuous_actions_row, item)| {
                        if let Some(ref input_buffer) = item.input_buffer {
                            continuous_actions_row
                                .copy_from_slice(&input_buffer.history_continuous_actions);
                        }
                    });
            }

            history_product_surfaces_slice
                .par_chunks_exact_mut(history_seq_len)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(product_surface_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        product_surface_row.copy_from_slice(&input_buffer.history_product_surfaces);
                    }
                });

            let n_user_cat = model_config.hash_table.num_user_categorical_features;
            if n_user_cat > 0
                && let Some(sl) = user_categorical_slice
            {
                sl.par_chunks_exact_mut(n_user_cat)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.user_categorical_features);
                        }
                    });
            }

            let n_user_bool = model_config.hash_table.num_user_bool_features;
            if n_user_bool > 0
                && let Some(sl) = user_bool_slice
            {
                sl.par_chunks_exact_mut(n_user_bool)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.user_bool_features);
                        }
                    });
            }
            let n_user_float = model_config.hash_table.num_user_float_features;
            if n_user_float > 0
                && let Some(sl) = user_float_slice
            {
                sl.par_chunks_exact_mut(n_user_float)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.user_float_features);
                        }
                    });
            }
            let n_user_int64 = model_config.hash_table.num_user_int64_features;
            if n_user_int64 > 0
                && let Some(sl) = user_int64_slice
            {
                sl.par_chunks_exact_mut(n_user_int64)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.user_int64_features);
                        }
                    });
            }

            let n_user_installed_apps = model_config.hash_table.num_user_installed_apps;
            if n_user_installed_apps > 0
                && let Some(sl) = user_installed_apps_slice
            {
                sl.par_chunks_exact_mut(n_user_installed_apps)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.user_installed_apps);
                        }
                    });
            }

            let n_post_cat = model_config.hash_table.num_post_categorical_features;
            if n_post_cat > 0
                && let Some(sl) = hist_cat_slice
            {
                sl.par_chunks_exact_mut(history_seq_len * n_post_cat)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.history_categorical_features);
                        }
                    });
            }
            let n_post_bool = model_config.hash_table.num_post_bool_features;
            if n_post_bool > 0
                && let Some(sl) = hist_bool_slice
            {
                sl.par_chunks_exact_mut(history_seq_len * n_post_bool)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.history_bool_features);
                        }
                    });
            }
            let n_post_float = model_config.hash_table.num_post_float_features;
            if n_post_float > 0
                && let Some(sl) = hist_float_slice
            {
                sl.par_chunks_exact_mut(history_seq_len * n_post_float)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.history_float_features);
                        }
                    });
            }
            let n_post_int64 = model_config.hash_table.num_post_int64_features;
            if n_post_int64 > 0
                && let Some(sl) = hist_int64_slice
            {
                sl.par_chunks_exact_mut(history_seq_len * n_post_int64)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.history_int64_features);
                        }
                    });
            }

            history_post_ids_slice
                .par_chunks_exact_mut(history_seq_len)
                .take(length_of_input)
                .zip(request.items.par_iter())
                .for_each(|(post_id_row, item)| {
                    if let Some(ref input_buffer) = item.input_buffer {
                        post_id_row.copy_from_slice(&input_buffer.history_post_ids);
                    }
                });

            let sid_dim = model_config.sid_num_levels;
            if sid_dim > 0
                && let Some(ref mut sl) = hist_sid_slice
            {
                sl.par_chunks_exact_mut(history_seq_len * sid_dim)
                    .take(length_of_input)
                    .zip(request.items.par_iter())
                    .for_each(|(row, item)| {
                        if let Some(ref buf) = item.input_buffer {
                            row.copy_from_slice(&buf.history_semantic_ids);
                        }
                    });
            }
        });

        Ok(true)
    }
}

struct RecsysRetrievalPredictorImpl {
    sender: mpsc::Sender<RetrieveRequestItem>,
    gauge: LoadGauge,
    inflight_semaphore: Semaphore,
    model_config: ModelConfig,
    #[allow(dead_code)]
    reload_request: Arc<AtomicBool>,
    checkpoint_loading: Arc<AtomicBool>,
    current_checkpoint_timestamp: Arc<AtomicU64>,
    serving_prefix: Arc<StdMutex<String>>,
    peer_ready: Arc<AtomicBool>,
    peer_download_enabled: bool,
    reload_directive: Arc<StdMutex<Option<ReloadDirective>>>,
    enqueue_timeout_ms: u64,
    mm_embeddings_client: Option<MmEmbeddingsClient>,
    sid_client: Option<Arc<crate::sid_client::SemanticIdClient>>,
    admission: Arc<AdmissionController>,
    prefetch_mm_query_config: Option<Arc<PrefetchMmQueryConfig>>,
}

impl RecsysRetrievalPredictorImpl {
    async fn retrieve_top_k_candidates_inner(
        &self,
        request: Request<pb::RetrieveTopKCandidatesRequest>,
        start: Instant,
    ) -> tonic::Result<Response<pb::RetrieveTopKCandidatesResponse>> {
        let cancel_token = CancellationToken::new();
        let cancel_guard: DropGuard = cancel_token.clone().drop_guard();
        let (complete_tx, complete_rx) = oneshot::channel::<()>();
        let grpc_timeout = extract_grpc_timeout(&request);
        record_grpc_timeout_presence(grpc_timeout);
        let deadline = grpc_timeout.map(|t| start + t);
        let client = client_label(&request);
        record_client_server_network_delay(&request, &client);
        let mut request = request.into_inner();
        NUM_REQUESTS_RECEIVED.inc();
        record_not_ready_arrival(&client);

        if self.checkpoint_loading.load(Ordering::SeqCst) {
            NUM_REQUESTS_REJECTED
                .with_label_values(&["checkpoint_loading", client.as_str()])
                .inc();
            return Err(Status::unavailable(
                "Server is loading checkpoint, please retry on another pod",
            ));
        }

        let permit = self.inflight_semaphore.try_acquire().map_err(|_| {
            NUM_REQUESTS_REJECTED
                .with_label_values(&["inflight_cap", client.as_str()])
                .inc();
            Status::resource_exhausted(
                "Maximum concurrent requests limit reached on recsys server.",
            )
        })?;

        let elapsed_us = start.elapsed().as_micros() as u64;
        if let Some(remaining_us) = remaining_deadline_budget_us(
            grpc_timeout,
            self.admission.fallback_budget_us(),
            elapsed_us,
        ) {
            ADMISSION_BUDGET_REMAINING
                .with_label_values(&[client.as_str()])
                .observe(remaining_us as f64 / 1000.0);
        }
        if let Some(reason) =
            should_shed_for_deadline(&self.admission, &self.sender, grpc_timeout, elapsed_us)
        {
            DEADLINE_SHED.with_label_values(&[client.as_str()]).inc();
            NUM_REQUESTS_REJECTED
                .with_label_values(&["deadline_admission", client.as_str()])
                .inc();
            log::warn!("Retrieval request shed at admission: {}", reason);
            return Err(Status::resource_exhausted(reason));
        }

        let guard = self
            .gauge
            .try_acquire(request.load_ticket_id, 1, None)
            .ok_or_else(|| {
                NUM_REQUESTS_REJECTED
                    .with_label_values(&["lb_ticket", client.as_str()])
                    .inc();
                Status::resource_exhausted("Unable to acquire loadbalancer ticket")
            })?;
        let _in_flight_guard = InFlightGuard::new();

        let user_id = request.user_ids.pop().unwrap();
        let sequence = request.sequences.pop();
        let topic_entity_ids = request.topic_entity_ids;
        let topic_filter_mode = request.topic_filter_mode.unwrap_or(0);

        let columnar_sequence_bytes = request.columnar_sequences.pop().filter(|b| !b.is_empty());
        let eligible_posts_bloom_filter = request.eligible_posts_bloom_filter;

        let cancel_token_clone = cancel_token.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = cancel_token_clone.cancelled() => {
                    NUM_REQUESTS_CANCELED.inc();
                    warn!("Retrieval request was cancelled");
                }
                _ = complete_rx => {
                }
            }
        });

        let (mm_query_embeddings, mm_query_seed_post_ids) =
            match (&self.prefetch_mm_query_config, &self.mm_embeddings_client) {
                (Some(cfg), Some(client)) => {
                    crate::multimodal_retrieval::prefetch_seed_embeddings(
                        cfg,
                        client,
                        columnar_sequence_bytes.as_deref(),
                        sequence.as_ref(),
                    )
                    .await
                }
                _ => (None, None),
            };

        let top_k_candidates_result = handle_retrieval_request(
            user_id,
            sequence,
            columnar_sequence_bytes,
            request.top_k,
            topic_entity_ids,
            topic_filter_mode,
            eligible_posts_bloom_filter.to_vec(),
            &self.sender,
            &self.model_config,
            &cancel_token,
            self.enqueue_timeout_ms,
            &self.sid_client,
            request.client_context,
            request.user_context,
            deadline,
            mm_query_embeddings,
            mm_query_seed_post_ids,
        )
        .await;

        let top_k_candidates = match top_k_candidates_result {
            Ok(candidates) => vec![candidates],
            Err(e) => {
                if e.code() == tonic::Code::Cancelled {
                    log::warn!("Retrieval request cancelled with error: {:?}", e);
                } else {
                    NUM_REQUESTS_FAILED.inc();
                    log::error!("Retrieval request failed with error: {:?}", e);
                }
                cancel_guard.disarm();
                return Err(e);
            }
        };

        drop(guard);
        drop(permit);
        NUM_REQUESTS_SUCCEEDED.inc();
        REQUEST_LATENCY.observe(start.elapsed().as_secs_f64() * 1000.0);
        cancel_guard.disarm();
        let _ = complete_tx.send(());
        let retrieve_top_k_candidates_response =
            pb::RetrieveTopKCandidatesResponse { top_k_candidates };
        Ok(Response::new(retrieve_top_k_candidates_response))
    }
}

#[tonic::async_trait]
impl pb::recsys_retrieval_predictor_server::RecsysRetrievalPredictor
    for RecsysRetrievalPredictorImpl
{
    async fn retrieve_top_k_candidates(
        &self,
        request: Request<pb::RetrieveTopKCandidatesRequest>,
    ) -> tonic::Result<Response<pb::RetrieveTopKCandidatesResponse>> {
        let start = Instant::now();
        let client = client_label(&request);
        let workload = workload_label(&request);
        let src_dc = src_dc_label(&request);
        let result = self.retrieve_top_k_candidates_inner(request, start).await;
        request_metrics::emit_request_metrics(
            start,
            status_code(&result),
            &client,
            &workload,
            &src_dc,
        );
        result
    }

    async fn reload_model(
        &self,
        request: Request<pb::ReloadModelRequest>,
    ) -> tonic::Result<Response<pb::ReloadModelResponse>> {
        let req = request.into_inner();
        if req.do_reload {
            log::info!("ReloadModel RPC called with do_reload=true - setting reload_request flag");
            if self.peer_download_enabled {
                *self.reload_directive.lock().unwrap() = if req.copy_urls.is_empty() {
                    None
                } else {
                    log::info!(
                        "ReloadModel P2P directive: {} peer url(s), target_prefix={:?}",
                        req.copy_urls.len(),
                        req.target_prefix
                    );
                    Some((req.copy_urls.clone(), req.target_prefix.clone()))
                };
            }
            self.reload_request.store(true, Ordering::SeqCst);
        } else {
            log::info!("ReloadModel RPC called with do_reload=false - poll mode, not reloading");
        }
        Ok(Response::new(pb::ReloadModelResponse {
            reload_in_progress: self.reload_request.load(Ordering::SeqCst),
            current_checkpoint_timestamp: f64::from_bits(
                self.current_checkpoint_timestamp.load(Ordering::SeqCst),
            ),
            serving_prefix: self.serving_prefix.lock().unwrap().clone(),
            peer_ready: self.peer_ready.load(Ordering::SeqCst),
        }))
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_retrieval_request(
    user_id: u64,
    sequence: Option<UserActionSequence>,
    columnar_sequence_bytes: Option<Bytes>,
    top_k: u32,
    topic_entity_ids: Vec<u64>,
    topic_filter_mode: i32,
    eligible_posts_bloom_filter: Vec<u8>,
    sender: &mpsc::Sender<RetrieveRequestItem>,
    model_config: &ModelConfig,
    cancellation_token: &CancellationToken,
    enqueue_timeout_ms: u64,
    sid_client: &Option<Arc<crate::sid_client::SemanticIdClient>>,
    client_context: Option<pb::ClientContext>,
    user_context: Option<pb::UserContext>,
    deadline: Option<Instant>,
    mm_query_embeddings: Option<Vec<f16>>,
    mm_query_seed_post_ids: Option<Vec<u64>>,
) -> tonic::Result<ScoredCandidates> {
    let (tx, rx) = oneshot::channel();

    let ip_address = client_context
        .as_ref()
        .map(|ctx| ctx.ip_address.as_str())
        .unwrap_or("");

    let start = std::time::Instant::now();
    let mut input_buffer = if let Some(ref columnar_bytes) = columnar_sequence_bytes {
        log::debug!(
            "Computing retrieval input buffer from columnar bytes ({} bytes)",
            columnar_bytes.len()
        );

        match RetrievalInputBuffer::compute_from_columnar_bytes(
            model_config,
            columnar_bytes,
            user_id,
            ip_address,
            client_context.as_ref(),
            user_context.as_ref(),
        ) {
            Ok(buf) => buf,
            Err(e) => {
                log::error!(
                    "Failed to parse columnar bytes for retrieval, falling back to proto path: {}",
                    e
                );
                RetrievalInputBuffer::compute_for_item(
                    model_config,
                    &sequence,
                    ip_address,
                    client_context.as_ref(),
                    user_context.as_ref(),
                )
            }
        }
    } else {
        RetrievalInputBuffer::compute_for_item(
            model_config,
            &sequence,
            ip_address,
            client_context.as_ref(),
            user_context.as_ref(),
        )
    };

    let sid_num_levels = model_config.sid_num_levels;
    if sid_num_levels > 0
        && let Some(client) = sid_client.as_ref()
    {
        let history_ids: Vec<i64> = input_buffer
            .history_post_ids
            .iter()
            .copied()
            .filter(|&id| id != 0)
            .collect();
        let sids = client.lookup(&history_ids).await;
        fill_semantic_ids(
            &sids,
            &input_buffer.history_post_ids,
            &mut input_buffer.history_semantic_ids,
            sid_num_levels,
        );
    }

    let duration = start.elapsed();
    INPUT_BUFFER_COMPUTATION_TIME.observe(duration.as_secs_f64());

    let item = RetrieveRequestItem {
        user_id,
        sequence,
        top_k,
        topic_entity_ids,
        topic_filter_mode,
        sender: Some(tx),
        input_buffer: Some(input_buffer),
        enqueue_time: Instant::now(),
        deadline,
        columnar_sequence_bytes,
        eligible_posts_bloom_filter,
        mm_query_embeddings,
        mm_query_seed_post_ids,
    };

    tokio::select! {
        res = sender.send(item) => {
            res.map_err(|_| Status::internal("Failed to send item to worker"))?;
        }
        _ = tokio::time::sleep(Duration::from_millis(enqueue_timeout_ms)) => {
            BATCHING_TIMEOUT_DROP.inc();
            return Err(Status::internal("Request timed out while waiting to enqueue"));
        }
        _ = cancellation_token.cancelled() => {
            log::warn!("Retrieval request was cancelled during batching");
            return Err(Status::cancelled("Request was cancelled"));
        }
    }

    tokio::select! {
        result = rx => match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(status)) => Err(status),
            Err(_recv_err) => {
                log::error!("Failed to receive result from worker for retrieval request");
                Err(Status::internal("Failed to receive result from worker"))
            }
        },
        _ = cancellation_token.cancelled() => {
            log::warn!("Retrieval request was cancelled while waiting for worker result");
            Err(Status::cancelled("Request was cancelled"))
        },
    }
}

macro_rules! server_impl {
    ($class_name:ident, $item_type:ty, $batch_type:ident, $service_struct:ident, $proto_server_type:ty, $prep_type:ty) => {
        #[pyclass(module = "xai_recsys_engine")]
        struct $class_name {
            runtime: tokio::runtime::Runtime,
            server: HttpServer,
            health_reporter: tonic_health::server::HealthReporter,
            batch_size: usize,
            model_config: ModelConfig,
            request_queue: RequestQueue<$item_type>,
                                    sender: mpsc::Sender<$item_type>,
            reload_request: Arc<AtomicBool>,
            checkpoint_loading: Arc<AtomicBool>,
            current_checkpoint_timestamp: Arc<AtomicU64>,
            serving_prefix: Arc<StdMutex<String>>,
            peer_ready: Arc<AtomicBool>,
            #[cfg(target_os = "linux")]
            copy_sender: Arc<Sender>,
            peer_serve_enabled: bool,
            reload_directive: Arc<StdMutex<Option<ReloadDirective>>>,
            admission: Arc<AdmissionController>,
        }

        #[pymethods]
        impl $class_name {
                    #[rustfmt::skip]
                    #[new]
                    #[pyo3(signature = (
                        grpc_port,
                        metrics_port,
                        batch_size,
                        max_inflight_requests,
                        *,
                        channel_size = CHANNEL_SIZE,
                        enqueue_timeout_ms = ENQUEUE_TIMEOUT_MS,
                        queue_max_staleness_ms = QUEUE_MAX_STALENESS_MS,
                        mm_client = None,
                        sid_client = None,
                        user_id_table_size = 100_000,
                        user_hash_scales = vec![196742702, 1852108266],
                        user_biases = vec![1935840681, 167407236],
                        user_modulus = 2_859_568_897,
                        item_id_table_size = 100_000,
                        item_hash_vocab_size = 100_000,
                        item_hash_scales = vec![2161410491, 1754358832],
                        item_biases = vec![1935840681, 167407236],
                        item_modulus = 2_361_375_383,
                        author_id_table_size = 10_000,
                        author_hash_scales = vec![371965780, 328930218],
                        author_biases = vec![139686260, 37755056],
                        author_modulus = 631_860_353,
                        output_vocab_size = 64,
                        num_continuous_actions = 8,
                        history_seq_len = 1000,
                        candidate_seq_len = 1400,
                        multimodal_embedding_dim = 0,
                        search_query_embedding_dim = 0,
                        ip_id_table_size = 0,
                        ip_hash_scales = vec![529_482_163, 1_327_604_891],
                        ip_biases = vec![742_961_053, 318_205_477],
                        ip_modulus = 1_073_741_789,
                        num_categorical_features = 0,
                        num_user_categorical_features = 0,
                        num_user_bool_features = 0,
                        num_user_float_features = 0,
                        num_user_int64_features = 0,
                        num_user_installed_apps = 0,
                        num_post_categorical_features = 0,
                        num_post_bool_features = 0,
                        num_post_float_features = 0,
                        num_post_int64_features = 0,
                        enable_stale_post = false,
                        enable_async_response_compression = false,
                        sid_num_levels = 0,
                        copy_max_entries = DEFAULT_MAX_ENTRIES,
                        peer_serve_enabled = false,
                        peer_serve_rate_limit_gib_per_s = 0.0,
                        peer_serve_max_concurrent_sends = 4_usize,
                        peer_download_enabled = false,
                        enable_deadline_admission = false,
                        deadline_margin_ms = 30.0,
                        deadline_post_ms = 2.0,
                        deadline_fallback_budget_ms = 1200.0,
                        service_time_ewma_alpha = 0.2,
                        pipeline_depth = 1,
                        prefetch_mm_query_for_retrieval = false,
                        mm_query_sample_size = 0_usize,
                        mm_query_max_age_hours = 48_u64,
                        mm_query_positive_action_types = Vec::<i32>::new(),
                        tls_cert_path = None,
                        tls_key_path = None,
                        tls_client_ca_path = None,
                    ))]
            fn new(
                        grpc_port: u16,
                        metrics_port: u16,
                        batch_size: usize,
                        max_inflight_requests: usize,
                        channel_size: usize,
                        enqueue_timeout_ms: u64,
                        queue_max_staleness_ms: u64,
                        mm_client: Option<&PyMmEmbeddingsClient>,
                        sid_client: Option<&crate::sid_client::PySemanticIdClient>,
                        user_id_table_size: usize,
                        user_hash_scales: Vec<i64>,
                        user_biases: Vec<i64>,
                        user_modulus: i64,
                        item_id_table_size: usize,
                        item_hash_vocab_size: usize,
                        item_hash_scales: Vec<i64>,
                        item_biases: Vec<i64>,
                        item_modulus: i64,
                        author_id_table_size: usize,
                        author_hash_scales: Vec<i64>,
                        author_biases: Vec<i64>,
                        author_modulus: i64,
                        output_vocab_size: usize,
                        num_continuous_actions: usize,
                        history_seq_len: usize,
                        candidate_seq_len: usize,
                        multimodal_embedding_dim: usize,
                        search_query_embedding_dim: usize,
                        ip_id_table_size: usize,
                        ip_hash_scales: Vec<i64>,
                        ip_biases: Vec<i64>,
                        ip_modulus: i64,
                        num_categorical_features: usize,
                        num_user_categorical_features: usize,
                        num_user_bool_features: usize,
                        num_user_float_features: usize,
                        num_user_int64_features: usize,
                        num_user_installed_apps: usize,
                        num_post_categorical_features: usize,
                        num_post_bool_features: usize,
                        num_post_float_features: usize,
                        num_post_int64_features: usize,
                        enable_stale_post: bool,
                        enable_async_response_compression: bool,
                        sid_num_levels: usize,
                        #[allow(unused_variables)]
                        copy_max_entries: usize,
                        peer_serve_enabled: bool,
                        #[allow(unused_variables)]
                        peer_serve_rate_limit_gib_per_s: f64,
                        #[allow(unused_variables)]
                        peer_serve_max_concurrent_sends: usize,
                        peer_download_enabled: bool,
                        enable_deadline_admission: bool,
                        deadline_margin_ms: f64,
                        deadline_post_ms: f64,
                        deadline_fallback_budget_ms: f64,
                        service_time_ewma_alpha: f64,
                        pipeline_depth: u64,
                        prefetch_mm_query_for_retrieval: bool,
                        mm_query_sample_size: usize,
                        mm_query_max_age_hours: u64,
                        mm_query_positive_action_types: Vec<i32>,
                        tls_cert_path: Option<String>,
                        tls_key_path: Option<String>,
                        tls_client_ca_path: Option<String>,
                    ) -> PyResult<Self> {
                        xai_recsys_server::init_env_logger();
                        let queue_max_staleness = if queue_max_staleness_ms == 0 {
                            Duration::MAX
                        } else {
                            Duration::from_millis(queue_max_staleness_ms)
                        };
                        let (request_queue, sender, background_task) =
                            RequestQueue::<$item_type>::new(
                                batch_size,
                                channel_size,
                                queue_max_staleness,
                            );

                        let admission = Arc::new(AdmissionController::new(AdmissionConfig {
                            enabled: enable_deadline_admission,
                            batch_size: batch_size as u64,
                            margin_us: (deadline_margin_ms * 1000.0) as u64,
                            post_us: (deadline_post_ms * 1000.0) as u64,
                            fallback_budget_us: (deadline_fallback_budget_ms * 1000.0) as u64,
                            ewma_alpha: service_time_ewma_alpha,
                            pipeline_depth,
                            eta_model: AdmissionEtaModel::Loop,
                        }));

                        let gauge = LoadGauge::default();
                        let inflight_semaphore = Semaphore::new(max_inflight_requests);

                        let mut internal_model_config = ModelConfig::from_params(
                            user_id_table_size,
                            user_hash_scales,
                            user_biases,
                            user_modulus,
                            item_id_table_size,
                            item_hash_vocab_size,
                            item_hash_scales,
                            item_biases,
                            item_modulus,
                            author_id_table_size,
                            author_hash_scales,
                            author_biases,
                            author_modulus,
                            output_vocab_size,
                            num_continuous_actions,
                            history_seq_len,
                            candidate_seq_len,
                            multimodal_embedding_dim,
                            search_query_embedding_dim,
                            ip_id_table_size,
                            ip_hash_scales,
                            ip_biases,
                            ip_modulus,
                            num_categorical_features,
                            num_user_categorical_features,
                            num_user_bool_features,
                            num_user_float_features,
                            num_user_int64_features,
                            num_user_installed_apps,
                            num_post_categorical_features,
                            num_post_bool_features,
                            num_post_float_features,
                            num_post_int64_features,
                            enable_stale_post,
                        );
                        internal_model_config.sid_num_levels = sid_num_levels;
                        let reload_request = Arc::new(AtomicBool::new(false));
                        let checkpoint_loading = Arc::new(AtomicBool::new(false));
                        let current_checkpoint_timestamp = Arc::new(AtomicU64::new(0));
                        let serving_prefix = Arc::new(StdMutex::new(String::new()));
                        let peer_ready = Arc::new(AtomicBool::new(false));
                        let reload_directive = Arc::new(StdMutex::new(None));
                        #[cfg(target_os = "linux")]
                        let copy_sender = Arc::new(if peer_serve_enabled {
                            Sender::new(copy_max_entries).with_peer_serve(
                                peer_serve_max_concurrent_sends,
                                (peer_serve_rate_limit_gib_per_s > 0.0).then(|| {
                                    peer_serve_rate_limit_gib_per_s * (1u64 << 30) as f64
                                }),
                            )
                        } else {
                            Sender::new(copy_max_entries)
                        });

                        let runtime = tokio::runtime::Builder::new_multi_thread()
                            .enable_all()
                            .worker_threads(num_cpus::get().clamp(1, 64))
                            .build()
                            .unwrap();
                        let _guard = runtime.enter();

                        let mm_embeddings_client = mm_client.map(|c| c.client().clone());
                        let sid_client_arc = sid_client.map(|c| c.build());

                        let prefetch_mm_query_config: Option<Arc<PrefetchMmQueryConfig>> =
                            if prefetch_mm_query_for_retrieval && mm_embeddings_client.is_some() {
                                if mm_query_sample_size == 0 || multimodal_embedding_dim == 0 {
                                    log::warn!(
                                        "prefetch_mm_query_for_retrieval=true requires mm_query_sample_size > 0 and multimodal_embedding_dim > 0 (got {} / {}); disabling pre-fetch",
                                        mm_query_sample_size,
                                        multimodal_embedding_dim,
                                    );
                                    None
                                } else {
                                    let positive_set: std::collections::HashSet<i32> =
                                        mm_query_positive_action_types.into_iter().collect();
                                    Some(Arc::new(PrefetchMmQueryConfig {
                                        mm_embedding_dim: multimodal_embedding_dim,
                                        query_sample_size: mm_query_sample_size,
                                        max_age_hours: mm_query_max_age_hours,
                                        positive_action_types: Arc::new(positive_set),
                                    }))
                                }
                            } else {
                                None
                            };

                        let service = $service_struct {
                            sender: sender.clone(),
                            gauge: gauge.clone(),
                            inflight_semaphore,
                            model_config: internal_model_config.clone(),
                            reload_request: reload_request.clone(),
                            checkpoint_loading: checkpoint_loading.clone(),
                            current_checkpoint_timestamp: current_checkpoint_timestamp.clone(),
                            serving_prefix: serving_prefix.clone(),
                            peer_ready: peer_ready.clone(),
                            peer_download_enabled,
                            reload_directive: reload_directive.clone(),
                            enqueue_timeout_ms,
                            mm_embeddings_client,
                            sid_client: sid_client_arc,
                            admission: admission.clone(),
                            prefetch_mm_query_config,
                        };

                        let reflection_service = tonic_reflection::server::Builder::configure()
                            .register_encoded_file_descriptor_set(pb::FILE_DESCRIPTOR_SET)
                            .build_v1()
                            .unwrap();
                        let (health_reporter, health_service) = tonic_health::server::health_reporter();
                        let mut grpc_routes = RoutesBuilder::default();
                        let grpc_service = <$proto_server_type>::new(service)
                            .max_decoding_message_size(MAX_MESSAGE_SIZE)
                            .max_encoding_message_size(MAX_MESSAGE_SIZE)
                            .accept_compressed(tonic::codec::CompressionEncoding::Zstd);
                        if enable_async_response_compression {
                            use crate::grpc_compression::GrpcCompressionService;
                            grpc_routes.add_service(GrpcCompressionService::new(grpc_service));
                        } else {
                            let grpc_service = grpc_service
                                .send_compressed(tonic::codec::CompressionEncoding::Zstd);
                            grpc_routes.add_service(grpc_service);
                        }
                        grpc_routes
                            .add_service(reflection_service)
                            .add_service(health_service);
                        #[cfg(target_os = "linux")]
                        grpc_routes.add_service(CopyService::from_arc(copy_sender.clone()));
                        let http_router = axum::Router::new().route(
                            "/autometrics",
                            get(|| async { prometheus_exporter::encode_http_response() }),
                        );

                        let tls_config = crate::tls::ServerTlsOptions::from_paths(
                            tls_cert_path,
                            tls_key_path,
                            tls_client_ca_path,
                        )?
                        .map(|o| o.load())
                        .transpose()?;

                        let server = runtime.block_on(async move {
                            let mut grpc_config = GrpcConfig::new(grpc_port, grpc_routes.routes())
                                .with_readyz_endpoint();
                            if let Some(tls) = tls_config {
                                grpc_config = grpc_config.with_tls_config(tls);
                            }
                            HttpServer::builder(
                                metrics_port,
                                http_router,
                                CancellationToken::new(),
                                Duration::from_secs(20),
                            )
                            .with_grpc(grpc_config)
                            .with_so_reuseaddr(true)
                            .build()
                            .await
                        })?;

                        runtime.spawn(background_task);

                        Ok(Self {
                            runtime,
                            server,
                            health_reporter,
                            batch_size,
                            model_config: internal_model_config,
                            request_queue,
                            sender,
                            reload_request,
                            checkpoint_loading,
                            current_checkpoint_timestamp,
                            serving_prefix,
                            peer_ready,
                            #[cfg(target_os = "linux")]
                            copy_sender,
                            peer_serve_enabled,
                            reload_directive,
                            admission,
                        })
                    }

                                                fn record_service_time_ms(&self, ms: f64) {
                if ms > 0.0 {
                    request_metrics::record_admission_service_sample(
                        &self.admission,
                        (ms * 1000.0) as u64,
                    );
                }
            }

                        fn set_deadline_admission_enabled(&self, enabled: bool) {
                self.admission.set_enabled(enabled);
            }

            fn dequeue(&mut self) -> $batch_type {
                QUEUE_BACKLOG.set(queue_backlog(&self.sender));
                let items = self.runtime.block_on(self.request_queue.dequeue());
                if !items.is_empty() {
                    let now = Instant::now();
                    let max_age_ms = items
                        .iter()
                        .map(|item| now.duration_since(item.enqueue_time).as_secs_f64() * 1000.0)
                        .fold(0.0_f64, f64::max);
                    request_metrics::ITEM_QUEUE_TIME.observe(max_age_ms);
                }
                $batch_type { items }
            }

            fn set_ready(&self, is_ready: bool) {
                if is_ready {
                    disarm_not_ready_probe();
                } else {
                    arm_not_ready_probe();
                }
                self.server.set_readiness(is_ready);
                let status = if is_ready {
                    tonic_health::ServingStatus::Serving
                } else {
                    tonic_health::ServingStatus::NotServing
                };
                self.runtime.block_on(
                    self.health_reporter.set_service_status("", status),
                );
            }

            fn is_finished(&self) -> bool {
                self.server.is_terminated()
            }

            fn cancel_forcefully(&self) {
                self.server.terminate_forcefully();
            }

            fn cancel(&self) {
                self.server.terminate_gracefully();
            }

            fn check_reload_request(&self) -> bool {
                self.reload_request.load(Ordering::SeqCst)
            }

            fn reset_reload_request(&self) {
                self.reload_request.store(false, Ordering::SeqCst);
            }

                                                fn take_reload_directive(&self) -> Option<(Vec<String>, String)> {
                self.reload_directive.lock().unwrap().take()
            }

                                                                        fn set_checkpoint_timestamp(&self, timestamp: f64) {
                self.current_checkpoint_timestamp
                    .store(timestamp.to_bits(), Ordering::SeqCst);
            }

                                    fn set_serving_prefix(&self, prefix: &str) {
                *self.serving_prefix.lock().unwrap() = prefix.to_string();
            }

                                                                                    #[cfg(target_os = "linux")]
            fn publish_peer_manifest(
                slf: PyRef<'_, Self>,
                prefix: &str,
                entries: Vec<(String, String, u64, u64)>,
                blobs: Vec<(String, Vec<u8>)>,
            ) -> PyResult<bool> {
                *slf.serving_prefix.lock().unwrap() = prefix.to_string();
                if !slf.peer_serve_enabled {
                    return Ok(false);
                }
                let specs = entries
                    .into_iter()
                    .map(|(name, file, offset, size)| ManifestSpec {
                        name,
                        file,
                        offset: offset as usize,
                        size: size as usize,
                    })
                    .collect();
                let py = slf.py();
                let copy_sender = slf.copy_sender.clone();
                let peer_ready = slf.peer_ready.clone();
                drop(slf);
                py.detach(move || copy_sender.publish_manifest(specs, blobs))
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "publish_peer_manifest: {e}"
                        ))
                    })?;
                peer_ready.store(true, Ordering::SeqCst);
                Ok(true)
            }

                        #[cfg(not(target_os = "linux"))]
            fn publish_peer_manifest(
                &self,
                _py: Python,
                prefix: &str,
                _entries: Vec<(String, String, u64, u64)>,
                _blobs: Vec<(String, Vec<u8>)>,
            ) -> PyResult<bool> {
                *self.serving_prefix.lock().unwrap() = prefix.to_string();
                if self.peer_serve_enabled {
                    log::warn!("peer_serve_enabled is unsupported on this platform");
                }
                Ok(false)
            }

                                                #[cfg(target_os = "linux")]
            #[pyo3(signature = (drain_timeout_secs = 10.0))]
            fn revoke_peer_manifest(slf: PyRef<'_, Self>, drain_timeout_secs: f64) -> bool {
                slf.peer_ready.store(false, Ordering::SeqCst);
                let py = slf.py();
                let copy_sender = slf.copy_sender.clone();
                drop(slf);
                py.detach(move || {
                    copy_sender
                        .revoke_manifest(Duration::from_secs_f64(drain_timeout_secs.max(0.0)))
                })
            }

            #[cfg(not(target_os = "linux"))]
            #[pyo3(signature = (drain_timeout_secs = 10.0))]
            fn revoke_peer_manifest(&self, _py: Python, drain_timeout_secs: f64) -> bool {
                let _ = drain_timeout_secs;
                self.peer_ready.store(false, Ordering::SeqCst);
                true
            }

                                                                        fn set_checkpoint_loading(&self) {
                self.checkpoint_loading.store(true, Ordering::SeqCst);
                arm_not_ready_probe();
                self.runtime.block_on(
                    self.health_reporter.set_service_status(
                        "",
                        tonic_health::ServingStatus::NotServing,
                    ),
                );
            }

                                                            fn clear_checkpoint_loading(&self) {
                self.checkpoint_loading.store(false, Ordering::SeqCst);
                disarm_not_ready_probe();
                self.runtime.block_on(
                    self.health_reporter.set_service_status(
                        "",
                        tonic_health::ServingStatus::Serving,
                    ),
                );
            }

                                                            fn reset_dequeue_timer(&mut self) {
                self.request_queue.reset_dequeue_timer();
            }

            #[pyo3(signature = (timeout=None))]
            fn wait(&mut self, timeout: Option<TimeDelta>) {
                let _guard = self.runtime.enter();
                self.runtime
                    .block_on(tokio::time::timeout(
                        timeout.unwrap_or(TimeDelta::MAX).to_std().unwrap(),
                        self.server.wait_for_termination(),
                    ))
                    .ok();
            }

            fn hash_user_id(&self, user_id: i64, index: usize) -> i32 {
                self.model_config.hash_table.hash_user_id(user_id, index)
            }

            fn hash_item_id(&self, item_id: i64, index: usize) -> i32 {
                self.model_config.hash_table.hash_item_id(item_id, index)
            }

            fn hash_author_id(&self, author_id: i64, index: usize) -> i32 {
                self.model_config
                    .hash_table
                    .hash_author_id(author_id, index)
            }

            fn hash_user_ids(&self, user_ids: Vec<i64>) -> Vec<i32> {
                self.model_config.hash_table.hash_user_ids(&user_ids)
            }

            fn hash_item_ids(&self, item_ids: Vec<i64>) -> Vec<i32> {
                self.model_config.hash_table.hash_item_ids(&item_ids)
            }

            fn hash_author_ids(&self, author_ids: Vec<i64>) -> Vec<i32> {
                self.model_config.hash_table.hash_author_ids(&author_ids)
            }

                                                                                                #[pyo3(signature = (request, request_in_flight_id, input, batch_size=None))]
            pub fn compute_batch_inputs_rust_quick<'py>(
                &self,
                py: Python<'py>,
                request: &$batch_type,
                request_in_flight_id: usize,
                input: &$prep_type,
                batch_size: Option<usize>,
            ) -> PyResult<bool> {
                input.compute_batch_inputs_rust_quick(
                    py,
                    &self.model_config,
                    request,
                    request_in_flight_id,
                    batch_size.unwrap_or(self.batch_size),
                )
            }
        }
    };
}

server_impl!(
    RecsysPredictorServer,
    PredictRequestItem,
    PredictRequestBatch,
    RecsysPredictorImpl,
    pb::recsys_predictor_server::RecsysPredictorServer<RecsysPredictorImpl>,
    RankingBatchPrep
);

server_impl!(
    RecsysRetrievalPredictorServer,
    RetrieveRequestItem,
    RetrieveRequestBatch,
    RecsysRetrievalPredictorImpl,
    pb::recsys_retrieval_predictor_server::RecsysRetrievalPredictorServer<
        RecsysRetrievalPredictorImpl,
    >,
    RetrievalBatchPrep
);

#[pymodule(name = "xai_recsys_engine")]
pub fn xai_recsys_engine(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    xai_recsys_server::init_env_logger();
    m.add_class::<RankingBatchPrep>()?;
    m.add_class::<RetrievalBatchPrep>()?;
    m.add_class::<PyMmEmbeddingsClient>()?;
    m.add_class::<crate::sid_client::PySemanticIdClient>()?;
    m.add_class::<RecsysPredictorServer>()?;
    m.add_class::<RecsysRetrievalPredictorServer>()?;
    m.add_class::<PredictRequestBatch>()?;
    m.add_class::<RetrieveRequestBatch>()?;
    m.add_class::<ReqUserSequenceAggrgated>()?;
    m.add_class::<ReqCandidateSequence>()?;
    m.add_function(wrap_pyfunction!(init_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(madvise_hugepage, m)?)?;
    m.add_function(wrap_pyfunction!(madvise_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(madvise_normal, m)?)?;
    {
        use crate::host_buffer::{EmbTableHandle, allocate_emb_table_py};
        m.add_function(wrap_pyfunction!(allocate_emb_table_py, m)?)?;
        m.add_class::<EmbTableHandle>()?;
    }
    m.add_function(wrap_pyfunction!(embedding_gather, m)?)?;
    m.add_function(wrap_pyfunction!(load_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(load_tensor_no_resharding, m)?)?;
    m.add_function(wrap_pyfunction!(maybe_load_fully_replicated_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(upload_files_to_storage, m)?)?;
    m.add_function(wrap_pyfunction!(download_prefix_to_local, m)?)?;
    m.add_function(wrap_pyfunction!(download_files_from_storage, m)?)?;
    m.add_function(wrap_pyfunction!(upload_inline_data_to_storage, m)?)?;
    m.add_function(wrap_pyfunction!(copy_chunks_to_storage, m)?)?;
    m.add_function(wrap_pyfunction!(reshard_col_to_row_and_upload, m)?)?;
    m.add_function(wrap_pyfunction!(load_reshard_row_to_col, m)?)?;
    m.add_function(wrap_pyfunction!(find_latest_storage_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(list_storage_objects, m)?)?;
    m.add_function(wrap_pyfunction!(load_arrays_from_storage, m)?)?;
    m.add_function(wrap_pyfunction!(remove_old_from_storage, m)?)?;
    m.add_function(wrap_pyfunction!(adler32_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(adler32_combine_py, m)?)?;
    m.add_function(wrap_pyfunction!(adler32_shard_partial, m)?)?;
    m.add_function(wrap_pyfunction!(adler32_combine_partials, m)?)?;
    Ok(())
}
