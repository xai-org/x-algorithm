// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
#[allow(unused_imports)]
use crate::mem_util::{
    copy_nontemporal, madvise_hugepage_internal, madvise_normal_internal,
    madvise_sequential_internal,
};

use crate::{
    adler32::adler32_combine,
    r#const::{LIST, SEND},
    copy_port_client::ProgressAt,
    grpc_util::{
        TRANSFER_FAILED_SENTINEL, block_on, block_on_rate_limited, freeze, get_bytes_mut,
        ready_call_parse,
    },
    proto, proto_parser,
    proto_parser::{
        BytesMutProtoExt, bytes, decode_names, encode_names, proto, repeated_bytes, repeated_ints,
    },
};
use futures::future::{BoxFuture, join_all};
use lazy_static::lazy_static;
use nix::{
    fcntl::{OFlag, open},
    sys::stat::Mode,
    sys::uio::pread,
    unistd::close,
};
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use prometheus::{Histogram, exponential_buckets, register_histogram};
use pyo3::PyResult;
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use rand::prelude::*;
use rayon::prelude::*;
use simd_adler32::adler32;
use static_assertions::const_assert;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vld4q_u64, vst4q_u64};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256i, _mm_sfence, _mm256_stream_load_si256, _mm256_stream_si256};
use std::{
    cmp,
    collections::{BTreeMap, HashMap},
    mem,
    os::fd::{AsFd, OwnedFd},
    slice,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant},
};
use tokio::runtime;
use tonic::{Status, transport};
#[cfg(target_os = "linux")]
use {
    crate::{
        copy::{Contexts, MRz, make_queues, rdma, register_memory_regions},
        ibverbs_util::make_ib_contexts,
    },
    std::sync::Arc,
};

#[cfg(target_arch = "aarch64")]
const PAGE_SIZE: usize = 2usize.pow(16);
#[cfg(not(target_arch = "aarch64"))]
const PAGE_SIZE: usize = 2usize.pow(12);

lazy_static! {
    static ref EMBEDDING_GATHER_DURATION_SECONDS: Histogram = register_histogram!(
        "recsys_engine_embedding_gather_duration_seconds",
        "Time spent inside embedding_gather (seconds).",
        exponential_buckets(0.00001, 1.4, 40).unwrap()
    )
    .unwrap();
}

#[pyfunction]
pub fn init_parallel<'py>(
    py: Python<'py>,
    mut arr: PyReadwriteArray1<'py, u8>,
    num_threads: usize,
) -> PyResult<()> {
    let slice = arr.as_slice_mut()?;
    if num_threads == 0 {
        return Err(PyValueError::new_err("num_threads must not be 0"));
    }
    py.detach(|| {
        let chunk_size = slice.len().div_ceil(num_threads);
        slice.par_chunks_mut(chunk_size).for_each(|chunk| {
            chunk
                .iter_mut()
                .step_by(PAGE_SIZE)
                .for_each(|elem| *elem = 0);
        });
    });
    Ok(())
}

#[pyfunction]
pub fn madvise_hugepage<'py>(
    _py: Python<'py>,
    mut arr: PyReadwriteArray1<'py, u8>,
) -> PyResult<()> {
    madvise_hugepage_internal(arr.as_slice_mut()?)?;
    Ok(())
}

#[pyfunction]
pub fn madvise_sequential<'py>(
    _py: Python<'py>,
    mut arr: PyReadwriteArray1<'py, u8>,
) -> PyResult<()> {
    madvise_sequential_internal(arr.as_slice_mut()?)?;
    Ok(())
}

#[pyfunction]
pub fn madvise_normal<'py>(_py: Python<'py>, mut arr: PyReadwriteArray1<'py, u8>) -> PyResult<()> {
    madvise_normal_internal(arr.as_slice_mut()?)?;
    Ok(())
}

fn copy_rows_generic(
    table_slice: &[u8],
    row_index_chunk: &[u32],
    row_chunk: &mut [u8],
    row_size: usize,
) {
    for (row_index, row) in row_index_chunk
        .iter()
        .zip(row_chunk.chunks_exact_mut(row_size))
    {
        let mut idx = (*row_index as usize) * row_size;
        if idx >= table_slice.len() {
            idx = 0;
        }
        row.copy_from_slice(&table_slice[idx..idx + row_size]);
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const ALIGNMENT: usize = 2usize.pow(6);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn copy_rows_neon(
    table_slice: &[u8],
    row_index_chunk: &[u32],
    row_chunk: &mut [u8],
    row_size: usize,
) {
    unsafe {
        for (row_index, row) in row_index_chunk
            .iter()
            .zip(row_chunk.chunks_exact_mut(row_size))
        {
            let mut idx = (*row_index as usize) * row_size;
            if idx >= table_slice.len() {
                idx = 0;
            }
            let mut i = 0;
            while i + 1024 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const u64;
                let b = row.as_mut_ptr().add(i) as *mut u64;
                vst4q_u64(b, vld4q_u64(a));
                vst4q_u64(b.add(8), vld4q_u64(a.add(8)));
                vst4q_u64(b.add(16), vld4q_u64(a.add(16)));
                vst4q_u64(b.add(24), vld4q_u64(a.add(24)));
                vst4q_u64(b.add(32), vld4q_u64(a.add(32)));
                vst4q_u64(b.add(40), vld4q_u64(a.add(40)));
                vst4q_u64(b.add(48), vld4q_u64(a.add(48)));
                vst4q_u64(b.add(56), vld4q_u64(a.add(56)));
                vst4q_u64(b.add(64), vld4q_u64(a.add(64)));
                vst4q_u64(b.add(72), vld4q_u64(a.add(72)));
                vst4q_u64(b.add(80), vld4q_u64(a.add(80)));
                vst4q_u64(b.add(88), vld4q_u64(a.add(88)));
                vst4q_u64(b.add(96), vld4q_u64(a.add(96)));
                vst4q_u64(b.add(104), vld4q_u64(a.add(104)));
                vst4q_u64(b.add(112), vld4q_u64(a.add(112)));
                vst4q_u64(b.add(120), vld4q_u64(a.add(120)));
                i += 1024;
            }
            if i + 512 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const u64;
                let b = row.as_mut_ptr().add(i) as *mut u64;
                vst4q_u64(b, vld4q_u64(a));
                vst4q_u64(b.add(8), vld4q_u64(a.add(8)));
                vst4q_u64(b.add(16), vld4q_u64(a.add(16)));
                vst4q_u64(b.add(24), vld4q_u64(a.add(24)));
                vst4q_u64(b.add(32), vld4q_u64(a.add(32)));
                vst4q_u64(b.add(40), vld4q_u64(a.add(40)));
                vst4q_u64(b.add(48), vld4q_u64(a.add(48)));
                vst4q_u64(b.add(56), vld4q_u64(a.add(56)));
                i += 512;
            }
            if i + 256 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const u64;
                let b = row.as_mut_ptr().add(i) as *mut u64;
                vst4q_u64(b, vld4q_u64(a));
                vst4q_u64(b.add(8), vld4q_u64(a.add(8)));
                vst4q_u64(b.add(16), vld4q_u64(a.add(16)));
                vst4q_u64(b.add(24), vld4q_u64(a.add(24)));
                i += 256;
            }
            if i + 128 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const u64;
                let b = row.as_mut_ptr().add(i) as *mut u64;
                vst4q_u64(b, vld4q_u64(a));
                vst4q_u64(b.add(8), vld4q_u64(a.add(8)));
                i += 128;
            }
            if i < row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const u64;
                let b = row.as_mut_ptr().add(i) as *mut u64;
                vst4q_u64(b, vld4q_u64(a));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_rows_avx2(
    table_slice: &[u8],
    row_index_chunk: &[u32],
    row_chunk: &mut [u8],
    row_size: usize,
) {
    unsafe {
        for (row_index, row) in row_index_chunk
            .iter()
            .zip(row_chunk.chunks_exact_mut(row_size))
        {
            let mut idx = (*row_index as usize) * row_size;
            if idx >= table_slice.len() {
                idx = 0;
            }
            let mut i = 0;
            while i + 512 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const __m256i;
                let b = row.as_mut_ptr().add(i) as *mut __m256i;
                _mm256_stream_si256(b, _mm256_stream_load_si256(a));
                _mm256_stream_si256(b.add(1), _mm256_stream_load_si256(a.add(1)));
                _mm256_stream_si256(b.add(2), _mm256_stream_load_si256(a.add(2)));
                _mm256_stream_si256(b.add(3), _mm256_stream_load_si256(a.add(3)));
                _mm256_stream_si256(b.add(4), _mm256_stream_load_si256(a.add(4)));
                _mm256_stream_si256(b.add(5), _mm256_stream_load_si256(a.add(5)));
                _mm256_stream_si256(b.add(6), _mm256_stream_load_si256(a.add(6)));
                _mm256_stream_si256(b.add(7), _mm256_stream_load_si256(a.add(7)));
                _mm256_stream_si256(b.add(8), _mm256_stream_load_si256(a.add(8)));
                _mm256_stream_si256(b.add(9), _mm256_stream_load_si256(a.add(9)));
                _mm256_stream_si256(b.add(10), _mm256_stream_load_si256(a.add(10)));
                _mm256_stream_si256(b.add(11), _mm256_stream_load_si256(a.add(11)));
                _mm256_stream_si256(b.add(12), _mm256_stream_load_si256(a.add(12)));
                _mm256_stream_si256(b.add(13), _mm256_stream_load_si256(a.add(13)));
                _mm256_stream_si256(b.add(14), _mm256_stream_load_si256(a.add(14)));
                _mm256_stream_si256(b.add(15), _mm256_stream_load_si256(a.add(15)));
                i += 512;
            }
            if i + 256 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const __m256i;
                let b = row.as_mut_ptr().add(i) as *mut __m256i;
                _mm256_stream_si256(b, _mm256_stream_load_si256(a));
                _mm256_stream_si256(b.add(1), _mm256_stream_load_si256(a.add(1)));
                _mm256_stream_si256(b.add(2), _mm256_stream_load_si256(a.add(2)));
                _mm256_stream_si256(b.add(3), _mm256_stream_load_si256(a.add(3)));
                _mm256_stream_si256(b.add(4), _mm256_stream_load_si256(a.add(4)));
                _mm256_stream_si256(b.add(5), _mm256_stream_load_si256(a.add(5)));
                _mm256_stream_si256(b.add(6), _mm256_stream_load_si256(a.add(6)));
                _mm256_stream_si256(b.add(7), _mm256_stream_load_si256(a.add(7)));
                i += 256;
            }
            if i + 128 <= row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const __m256i;
                let b = row.as_mut_ptr().add(i) as *mut __m256i;
                _mm256_stream_si256(b, _mm256_stream_load_si256(a));
                _mm256_stream_si256(b.add(1), _mm256_stream_load_si256(a.add(1)));
                _mm256_stream_si256(b.add(2), _mm256_stream_load_si256(a.add(2)));
                _mm256_stream_si256(b.add(3), _mm256_stream_load_si256(a.add(3)));
                i += 128;
            }
            if i < row_size {
                let a = table_slice.as_ptr().add(idx + i) as *const __m256i;
                let b = row.as_mut_ptr().add(i) as *mut __m256i;
                _mm256_stream_si256(b, _mm256_stream_load_si256(a));
                _mm256_stream_si256(b.add(1), _mm256_stream_load_si256(a.add(1)));
            }
        }
        _mm_sfence();
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn copy_rows(table_slice: &[u8], row_index_chunk: &[u32], row_chunk: &mut [u8], row_size: usize) {
    copy_rows_generic(table_slice, row_index_chunk, row_chunk, row_size)
}

#[cfg(target_arch = "aarch64")]
fn copy_rows(table_slice: &[u8], row_index_chunk: &[u32], row_chunk: &mut [u8], row_size: usize) {
    use std::arch::is_aarch64_feature_detected;
    if is_aarch64_feature_detected!("neon") {
        unsafe { copy_rows_neon(table_slice, row_index_chunk, row_chunk, row_size) }
    } else {
        copy_rows_generic(table_slice, row_index_chunk, row_chunk, row_size)
    }
}

#[cfg(target_arch = "x86_64")]
fn copy_rows(table_slice: &[u8], row_index_chunk: &[u32], row_chunk: &mut [u8], row_size: usize) {
    use std::is_x86_feature_detected;
    if is_x86_feature_detected!("avx2") {
        unsafe { copy_rows_avx2(table_slice, row_index_chunk, row_chunk, row_size) }
    } else {
        copy_rows_generic(table_slice, row_index_chunk, row_chunk, row_size)
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingGatherError(pub String);

impl std::fmt::Display for EmbeddingGatherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EmbeddingGatherError {}

pub fn embedding_gather_into(
    table: &[u8],
    row_indexes: &[u32],
    row_shards: &mut [&mut [u8]],
    row_size: usize,
    num_threads: usize,
) -> Result<(), EmbeddingGatherError> {
    let start = Instant::now();
    if row_size == 0 || !row_size.is_multiple_of(64) {
        return Err(EmbeddingGatherError(format!(
            "row_size {row_size} must be a non-zero multiple of 64"
        )));
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if !(table.as_ptr() as usize).is_multiple_of(ALIGNMENT) {
        return Err(EmbeddingGatherError(format!(
            "table must be {ALIGNMENT}-byte aligned"
        )));
    }
    let table_size = table.len();
    let num_row_indices = row_indexes.len();
    if !table_size.is_multiple_of(row_size) {
        return Err(EmbeddingGatherError(format!(
            "table size {table_size} must be divisible by row_size {row_size}"
        )));
    }
    if num_threads == 0 {
        return Err(EmbeddingGatherError(
            "num_threads must not be 0".to_string(),
        ));
    }
    let num_shards = row_shards.len();
    if num_shards == 0 || !num_row_indices.is_multiple_of(num_shards) {
        return Err(EmbeddingGatherError(format!(
            "row_indexes size {num_row_indices} must be divisible by the number of shards {num_shards}"
        )));
    }

    for row_slice in row_shards.iter() {
        let shard_output_capacity = row_slice.len();
        if shard_output_capacity * num_shards != num_row_indices * row_size {
            return Err(EmbeddingGatherError(format!(
                "shard_output_capacity {shard_output_capacity} must be row_indexes size {num_row_indices} times row_size {row_size} divided by the number of shards {num_shards}"
            )));
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        if !(row_slice.as_ptr() as usize).is_multiple_of(ALIGNMENT) {
            return Err(EmbeddingGatherError(format!(
                "shards must be {ALIGNMENT}-byte aligned"
            )));
        }
    }

    let chunk_size = num_row_indices / num_shards;
    row_indexes
        .par_chunks_exact(chunk_size)
        .zip(row_shards.par_iter_mut())
        .for_each(|(row_index_chunk, row_slice)| {
            let thread_chunk = row_index_chunk.len().div_ceil(num_threads);
            row_index_chunk
                .par_chunks(thread_chunk)
                .zip(row_slice.par_chunks_mut(row_size * thread_chunk))
                .for_each(|(row_index_chunk, row_chunk)| {
                    copy_rows(table, row_index_chunk, row_chunk, row_size);
                });
        });
    EMBEDDING_GATHER_DURATION_SECONDS.observe(start.elapsed().as_secs_f64());
    Ok(())
}

#[pyfunction]
pub fn embedding_gather<'py>(
    py: Python<'py>,
    table: PyReadonlyArray1<'py, u8>,
    row_indexes: PyReadonlyArray1<'py, u32>,
    rows: Bound<'py, PyTuple>,
    row_size: usize,
    num_threads: usize,
) -> PyResult<()> {
    let table_slice = table.as_slice()?;
    let row_index_slice = row_indexes.as_slice()?;

    let mut arrs: Vec<PyReadwriteArray1<'py, u8>> = Vec::with_capacity(rows.len());
    let err = || PyTypeError::new_err("rows must contain only 1-dim uint8 numpy arrays");
    for item in rows.iter() {
        arrs.push(item.extract().map_err(|_| err())?);
    }
    let mut row_slices: Vec<&mut [u8]> = Vec::with_capacity(arrs.len());
    for item in arrs.iter_mut() {
        row_slices.push(item.as_slice_mut()?);
    }

    py.detach(|| {
        embedding_gather_into(
            table_slice,
            row_index_slice,
            &mut row_slices,
            row_size,
            num_threads,
        )
    })
    .map_err(|e| PyValueError::new_err(e.0))
}

const READ_SIZE: usize = 2usize.pow(27);
const_assert!(READ_SIZE.is_multiple_of(PAGE_SIZE));
const WIDTH: usize = 2usize.pow(7);
const CONCURRENCY_OUTER: usize = 8;
const CONCURRENCY_INNER: usize = 16;

fn reassembly_slice_size(chunk_len: usize, row_size: usize) -> usize {
    cmp::max(1, chunk_len / row_size / CONCURRENCY_INNER) * row_size
}

lazy_static! {
    static ref CONNECT_TIMEOUT: Duration = {
        std::env::var("COPY_PORT_CONNECT_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(10))
    };
    static ref H2_CONN_WINDOW: Option<u32> = parse_window_mib(
        std::env::var("COPY_PORT_H2_CONN_WINDOW_MIB")
            .ok()
            .as_deref(),
        64,
    );
    static ref H2_STREAM_WINDOW: Option<u32> = parse_window_mib(
        std::env::var("COPY_PORT_H2_STREAM_WINDOW_MIB")
            .ok()
            .as_deref(),
        16,
    );
    static ref H2_ADAPTIVE_WINDOW: bool = matches!(
        std::env::var("COPY_PORT_H2_ADAPTIVE_WINDOW")
            .ok()
            .as_deref(),
        Some("1") | Some("true")
    );
}

fn parse_window_mib(v: Option<&str>, default_mib: u32) -> Option<u32> {
    let mib = v.and_then(|s| s.parse::<u32>().ok()).unwrap_or(default_mib);
    (mib > 0).then(|| mib.min(1024) << 20)
}

pub(crate) fn apply_copy_port_http2(endpoint: transport::Endpoint) -> transport::Endpoint {
    let mut endpoint = endpoint
        .initial_connection_window_size(*H2_CONN_WINDOW)
        .initial_stream_window_size(*H2_STREAM_WINDOW);
    if *H2_ADAPTIVE_WINDOW {
        endpoint = endpoint.http2_adaptive_window(true);
    }
    endpoint
}

pub(crate) async fn get_channels(target: String) -> Result<Vec<transport::Channel>, Status> {
    Ok(get_channels_with_endpoints(target)
        .await?
        .into_iter()
        .map(|(channel, _)| channel)
        .collect())
}

pub(crate) async fn get_channels_with_endpoints(
    target: String,
) -> Result<Vec<(transport::Channel, transport::Endpoint)>, Status> {
    if target.is_empty() {
        return Ok(Vec::new());
    }
    let tls = crate::tls::ClientTlsOptions::from_env()
        .map_err(|e| Status::invalid_argument(e.to_string()))?;
    let targets: Vec<_> = target.split(',').collect();
    let total = targets.len();
    let mut endpoints = Vec::with_capacity(total);
    for t in &targets {
        endpoints.push(
            crate::tls::endpoint_for(t, tls.as_ref())
                .map_err(|e| Status::invalid_argument(e.to_string()))?,
        );
    }
    let results = join_all(
        endpoints
            .into_iter()
            .zip(targets)
            .map(|(endpoint, target)| {
                let t = target.to_string();
                async move {
                    let endpoint =
                        apply_copy_port_http2(endpoint).connect_timeout(*CONNECT_TIMEOUT);
                    (
                        endpoint.connect().await.map(|channel| (channel, endpoint)),
                        t,
                    )
                }
            }),
    )
    .await;

    let mut failed = 0;
    let channels: Vec<_> = results
        .into_iter()
        .filter_map(|(result, target)| {
            if let Err(e) = result {
                log::error!("copy_port connect failed for {}: {}", target, e);
                failed += 1;
                return None;
            }
            result.ok()
        })
        .collect();

    if failed > 0 {
        log::warn!(
            "copy_port: connected to {}/{} channels ({} failed)",
            channels.len(),
            total,
            failed
        );
    } else {
        log::info!(
            "copy_port: connected to {}/{} channels",
            channels.len(),
            total
        );
    }
    Ok(channels)
}

pub(crate) fn peer_conns_per_source() -> usize {
    parse_conns_per_source(
        std::env::var("COPY_PORT_PEER_CONNS_PER_SOURCE")
            .ok()
            .as_deref(),
    )
}

pub(crate) fn trainer_conns_per_source() -> usize {
    parse_conns_per_source(
        std::env::var("COPY_PORT_TRAINER_CONNS_PER_SOURCE")
            .ok()
            .as_deref(),
    )
}

fn parse_conns_per_source(v: Option<&str>) -> usize {
    v.and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 16)
}

pub(crate) fn duplicated_urls(urls: &str, extra: usize) -> String {
    let one: Vec<&str> = urls
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let mut out = Vec::with_capacity(one.len() * extra);
    for _ in 0..extra {
        out.extend_from_slice(&one);
    }
    out.join(",")
}

pub(crate) async fn expand_replicated_channels(
    urls: &str,
    prefix: &str,
    channels: &mut Vec<transport::Channel>,
    conns_per_source: usize,
) {
    if conns_per_source <= 1 || channels.is_empty() {
        return;
    }
    let before = channels.len();
    let candidates = match get_channels(duplicated_urls(urls, conns_per_source - 1)).await {
        Ok(candidates) => candidates,
        Err(e) => {
            log::warn!(
                "copy_port: opening extra peer connections failed ({e}); skipping expansion"
            );
            return;
        }
    };
    let validated = join_all(candidates.into_iter().map(|channel| async move {
        match list_entries(slice::from_ref(&channel), prefix).await {
            Ok(lists) if lists.first().is_some_and(|l| !l.is_empty()) => Some(channel),
            Ok(_) => {
                log::warn!(
                    "copy_port: extra peer connection has no listing for \
                     {prefix}; dropping it from the stripe set"
                );
                None
            }
            Err(e) => {
                log::warn!(
                    "copy_port: extra peer connection listing for {prefix} \
                     failed ({e}); dropping it from the stripe set"
                );
                None
            }
        }
    }))
    .await;
    channels.extend(validated.into_iter().flatten());
    log::info!(
        "copy_port: replicated sources expanded from {} to {} connections \
         (conns_per_source={})",
        before,
        channels.len(),
        conns_per_source,
    );
}

pub(crate) struct JoinOnDrop<T> {
    handle: Option<tokio::task::JoinHandle<T>>,
}

impl<T> JoinOnDrop<T> {
    pub(crate) fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    pub(crate) async fn join(mut self) -> Result<T, tokio::task::JoinError> {
        let result = self.handle.as_mut().expect("join").await;
        self.handle.take();
        result
    }
}

struct JoinWake(std::thread::Thread);

impl std::task::Wake for JoinWake {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.0.unpark();
    }
}

impl<T> Drop for JoinOnDrop<T> {
    fn drop(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        let mut handle = tokio::task::unconstrained(handle);
        let thread = std::thread::current();
        let waker = std::task::Waker::from(std::sync::Arc::new(JoinWake(thread)));
        let mut context = std::task::Context::from_waker(&waker);
        while std::future::Future::poll(std::pin::Pin::new(&mut handle), &mut context).is_pending()
        {
            std::thread::park();
        }
    }
}

fn install_grpc_chunk(dest: &mut [u8], pos: &mut usize, checksum: &mut u32, chunk: &[u8]) {
    let n = chunk.len();
    if *pos + n <= dest.len() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            copy_nontemporal(&mut dest[*pos..*pos + n], chunk);
        }
        #[cfg(not(target_arch = "x86_64"))]
        dest[*pos..*pos + n].copy_from_slice(chunk);
    }
    *pos += n;
    adler32_combine(checksum, adler32(&chunk), n);
}

pub(crate) async fn send_entries(
    channel: transport::Channel,
    names: Vec<Vec<u8>>,
    offsets: Vec<usize>,
    sizes: Vec<usize>,
    buf: &mut [u8],
    label: String,
    #[cfg(target_os = "linux")] arg: (Vec<usize>, Contexts, MRz),
) -> (usize, u32) {
    send_entries_reporting(
        channel,
        names,
        offsets,
        sizes,
        buf,
        label,
        None,
        #[cfg(target_os = "linux")]
        arg,
    )
    .await
}

const PROGRESS_STEP: usize = 64 << 20;

fn report_written(progress: &ProgressAt, from: usize, to: usize) {
    if to > from {
        sfence_after_copy();
        progress
            .progress
            .written(progress.offset + from, progress.offset + to);
    }
}

fn sfence_after_copy() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_sfence();
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn send_entries_reporting(
    mut channel: transport::Channel,
    names: Vec<Vec<u8>>,
    offsets: Vec<usize>,
    sizes: Vec<usize>,
    buf: &mut [u8],
    label: String,
    progress: Option<ProgressAt>,
    #[cfg(target_os = "linux")] arg: (Vec<usize>, Contexts, MRz),
) -> (usize, u32) {
    #[cfg(target_os = "linux")]
    let (devices, contexts, mrz) = arg;

    let (all_names, prefix_sizes, suffix_sizes) = encode_names(names);
    let mut bytes = get_bytes_mut();
    bytes.put_string(1, &all_names);
    bytes.put_repeated_ints(
        [2, 3, 4, 5],
        [&prefix_sizes, &suffix_sizes, &offsets, &sizes],
    );
    #[cfg(target_os = "linux")]
    let Ok(queues) = make_queues(false, &devices, &contexts, &mrz, &mut bytes)
        .inspect_err(|e| log::error!("RDMA error: {e}"))
    else {
        return (TRANSFER_FAILED_SENTINEL, 0);
    };
    let body = freeze(bytes);

    const GRPC_COPY_PIECE: usize = 4 << 20;
    let dest_ptr = buf.as_mut_ptr() as usize;
    let dest_len = buf.len();
    let (tx, rx) = std::sync::mpsc::sync_channel::<bytes::Bytes>(2);
    let copy_progress = progress.clone();
    let copy_join = JoinOnDrop::new(tokio::task::spawn_blocking(move || {
        let dest = unsafe { std::slice::from_raw_parts_mut(dest_ptr as *mut u8, dest_len) };
        let mut pos = 0;
        let mut checksum = 1u32;
        let (mut landed, mut reported) = (0, 0);
        while let Ok(chunk) = rx.recv() {
            install_grpc_chunk(dest, &mut pos, &mut checksum, &chunk);
            if pos <= dest_len {
                landed = pos;
            }
            if let Some(progress) = &copy_progress
                && landed - reported >= PROGRESS_STEP
            {
                report_written(progress, reported, landed);
                reported = landed;
            }
        }
        if let Some(progress) = &copy_progress {
            report_written(progress, reported, landed);
        }
        (pos, checksum)
    }));
    let mut endpoints = Vec::new();
    let mut rmrs = Vec::new();
    let mut use_rdma = Vec::new();
    let proto = proto![
        (1, move |_, data: &[u8]| {
            let mut off = 0;
            while off < data.len() {
                let n = (data.len() - off).min(GRPC_COPY_PIECE);
                let piece = bytes::Bytes::copy_from_slice(&data[off..off + n]);
                if tx.send(piece).is_err() {
                    break;
                }
                off += n;
            }
        }),
        (2, repeated_bytes(&mut endpoints)),
        (3, repeated_bytes(&mut rmrs)),
        (4, proto_parser::bytes(&mut use_rdma)),
    ];
    if let Err(e) = ready_call_parse(SEND, proto, body, &mut channel, &label).await {
        log::error!(
            "gRPC error{suffix}: {e}",
            suffix = if label.is_empty() {
                String::new()
            } else {
                format!(" {label}")
            }
        );
        let _ = copy_join.join().await;
        return (TRANSFER_FAILED_SENTINEL, 0);
    }
    let Ok((pos, checksum)) = copy_join.join().await else {
        return (TRANSFER_FAILED_SENTINEL, 0);
    };

    use_rdma.resize(sizes.len(), 0);
    let f = |acc, (&x, &y)| if y != 0 { acc + x } else { acc };
    let size = sizes.iter().zip(&use_rdma).fold(0, f);
    let _used = &size;
    #[cfg(target_os = "linux")]
    match rdma(
        false, sizes, devices, contexts, mrz, queues, endpoints, rmrs, use_rdma,
    )
    .await
    {
        Ok(_) => {
            let Ok(checksum) = JoinOnDrop::new(tokio::task::spawn_blocking(move || {
                if size == 0 || pos >= dest_len {
                    return checksum;
                }
                let dest = unsafe { std::slice::from_raw_parts(dest_ptr as *const u8, dest_len) };
                let mut c = checksum;
                adler32_combine(&mut c, adler32(&&dest[pos..]), size);
                c
            }))
            .join()
            .await
            else {
                return (TRANSFER_FAILED_SENTINEL, 0);
            };
            if let Some(progress) = &progress
                && pos < dest_len
            {
                report_written(progress, pos, (pos + size).min(dest_len));
            }
            (pos + size, checksum)
        }
        Err(e) => {
            log::error!("RDMA error: {e}");
            (TRANSFER_FAILED_SENTINEL, 0)
        }
    }
    #[cfg(not(target_os = "linux"))]
    (pos, checksum)
}

pub(crate) async fn list_entries(
    channels: &[transport::Channel],
    prefix: &str,
) -> Result<Vec<Vec<(String, usize)>>, Status> {
    join_all(channels.iter().map(|channel| {
        let mut channel = channel.clone();
        async move {
            let body = freeze(get_bytes_mut());

            let mut names = Vec::new();
            let mut prefix_sizes = Vec::new();
            let mut suffix_sizes = Vec::new();
            let mut sizes = Vec::<usize>::new();
            let proto = proto![
                (1, bytes(&mut names)),
                (2, repeated_ints(&mut prefix_sizes)),
                (3, repeated_ints(&mut suffix_sizes)),
                (4, repeated_ints(&mut sizes)),
            ];
            ready_call_parse(LIST, proto, body, &mut channel, "").await?;

            if sizes.len() != suffix_sizes.len() {
                return Err(Status::invalid_argument(format!(
                    "bad size count: wanted {}, got {}",
                    suffix_sizes.len(),
                    sizes.len(),
                )));
            }
            Ok(decode_names(names, prefix_sizes, suffix_sizes)?
                .into_iter()
                .map(String::from_utf8)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Status::invalid_argument("UTF-8 error"))?
                .into_iter()
                .zip(sizes)
                .filter_map(|(x, y)| {
                    if x.starts_with(prefix) {
                        Some((x.get(prefix.len()..)?.to_string(), y))
                    } else {
                        None
                    }
                })
                .collect())
        }
    }))
    .await
    .into_iter()
    .collect()
}

pub(crate) async fn get_channels_list_entries(
    urls: String,
    prefix: String,
) -> Result<(Vec<transport::Channel>, Vec<Vec<(String, usize)>>), Status> {
    let channels = get_channels(urls).await?;
    let entries = list_entries(&channels[..], &prefix).await?;
    Ok((channels, entries))
}

#[pyfunction]
#[pyo3(signature = (elapsed_samples, urls, tensors, rate_limit_bytes_per_sec=None, max_concurrent_downloads=None, target_prefix=None, collect_entry_names=false))]
pub fn maybe_load_fully_replicated_tensors<'py>(
    py: Python<'py>,
    elapsed_samples: usize,
    urls: Bound<'py, PyString>,
    tensors: Bound<'py, PyList>,
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    target_prefix: Option<String>,
    collect_entry_names: bool,
) -> PyResult<Py<PyTuple>> {
    let runtime = runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let urls_string = urls.to_string();
    let (channels, entries) = py
        .detach(|| runtime.block_on(get_channels_list_entries(urls_string, String::new())))
        .map_err(|s| PyOSError::new_err(format!("gRPC error: {}", s.message())))?;

    if channels.is_empty() {
        return Err(PyOSError::new_err(
            "copy_port unreachable: no channels connected",
        ));
    }

    let mut counts = BTreeMap::<&str, usize>::new();
    for entries in &entries {
        let mut prev_prefix = "";
        for (name, _) in entries {
            if let Some(i) = name.find('/')
                && let Some(j) = name[i + 1..].find('/')
                && let prefix = &name[..i + j + 1]
                && prev_prefix != prefix
            {
                prev_prefix = prefix;
                let mut count = 1;
                if let Some(c) = counts.get(prefix) {
                    count += c;
                }
                counts.insert(prefix, count);
            }
        }
    }

    let target = target_prefix.as_deref().map(|t| t.trim_end_matches('/'));
    let mut prefix: &str = "";
    if let Some(target) = target {
        if counts.get(target) != Some(&channels.len()) {
            return Err(PyOSError::new_err(format!(
                "target prefix {target} not present on all {} channels",
                channels.len()
            )));
        }
        prefix = target;
    } else {
        for (name, count) in counts {
            if count == channels.len() {
                prefix = name;
            }
        }
    }
    let prev_prefix = format!("elapsed_samples_{elapsed_samples:018}/");
    if prefix.len() < prev_prefix.len() || prefix[..prev_prefix.len()] <= prev_prefix[..] {
        return Err(PyOSError::new_err(format!(
            "new prefix {prefix} is no greater than old prefix {prev_prefix}"
        )));
    }

    let number = rand::rng().random_range(0..channels.len());

    let mut names_sizes = HashMap::<&str, (&str, usize)>::new();
    let checksums_name = format!("{}/checksums.0.json", prefix);
    let mut checksums_size = 0;
    for &(ref name, size) in &entries[number] {
        let i = prefix.len() + 1;
        if name.starts_with(prefix)
            && name.len() > i
            && let Some(j) = name[i..].rfind("/c")
        {
            names_sizes.insert(&name[i..i + j], (name, size));
        } else if name == &checksums_name {
            checksums_size = size;
        }
    }

    let mut sizes = Vec::with_capacity(tensors.len());
    let mut futures = Vec::<BoxFuture<_>>::with_capacity(tensors.len());
    let mut entry_names = Vec::with_capacity(tensors.len());
    for tuple in tensors {
        let tuple = tuple
            .downcast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("type mismatch"))?;
        if tuple.len() != 2 {
            return Err(PyTypeError::new_err("type mismatch"));
        }
        let key: String = tuple
            .get_item(0)?
            .extract()
            .map_err(|_| PyTypeError::new_err("type mismatch"))?;
        let mut tensor: PyReadwriteArray1<'py, u8> = tuple
            .get_item(1)?
            .extract()
            .map_err(|_| PyTypeError::new_err("type mismatch"))?;
        let Some(&(name, size)) = names_sizes.get(&key[..]) else {
            return Err(PyValueError::new_err(format!(
                "cannot find {key} in {names_sizes:?}"
            )));
        };
        let sz = tensor.as_slice()?.len();
        if sz != size {
            return Err(PyValueError::new_err(format!(
                "size mismatch on {key}: listed {size}, requested {sz}"
            )));
        }
        sizes.push(size);
        if collect_entry_names {
            entry_names.push((key.clone(), name.to_string()));
        }
        let b: &'static mut [u8] = unsafe { mem::transmute(tensor.as_slice_mut()?) };
        futures.push(Box::pin(send_entries(
            channels[number].clone(),
            vec![name.as_bytes().to_vec()],
            vec![0],
            vec![size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        )));
    }

    let mut checksums = vec![0; checksums_size];
    if checksums_size != 0 {
        sizes.push(checksums_size);
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut checksums[..]) };
        futures.push(Box::pin(send_entries(
            channels[number].clone(),
            vec![checksums_name.as_bytes().to_vec()],
            vec![0],
            vec![checksums_size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        )));
    }

    let num_futures = futures.len();
    let results = py
        .detach(|| match rate_limit_bytes_per_sec {
            Some(limit) if limit > 0 => {
                let concurrent = max_concurrent_downloads.unwrap_or(num_futures.max(1));
                block_on_rate_limited(&runtime, futures, limit, concurrent)
            }
            _ => block_on(&runtime, futures, None),
        })
        .ok_or_else(|| PyOSError::new_err("copy_port timed out waiting for tensor download"))?;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
    for ((count, _), size) in results.into_iter().zip(sizes) {
        if count == TRANSFER_FAILED_SENTINEL {
            return Err(PyOSError::new_err(
                "copy_port gRPC/RDMA transfer failed (check Rust logs for details)",
            ));
        }
        if count != size {
            return Err(PyOSError::new_err(format!(
                "copy_port size mismatch: wanted {size} bytes, got {count}"
            )));
        }
    }

    let checksums_str = String::from_utf8(checksums).unwrap_or_default();

    let created_timestamp = serde_json::from_str::<serde_json::Value>(&checksums_str)
        .ok()
        .and_then(|v| v.get("created_timestamp")?.as_f64())
        .unwrap_or(0.0);

    if collect_entry_names {
        return Ok((prefix, checksums_str, created_timestamp, entry_names)
            .into_pyobject(py)?
            .into());
    }
    Ok((prefix, checksums_str, created_timestamp)
        .into_pyobject(py)?
        .into())
}

#[cfg(target_os = "linux")]
pub(crate) async fn register_tensor_for_download(
    shard_size: usize,
    num_channels: usize,
    step: usize,
    contexts: Contexts,
    devicez: Arc<Vec<Vec<usize>>>,
    buf: &mut [u8],
) -> Result<Vec<MRz>, Status> {
    let mut mrx = Vec::with_capacity(num_channels);
    let mut offset = 0;
    for i in 0..num_channels {
        mrx.push(Arc::new(
            register_memory_regions(
                &vec![shard_size; step],
                &devicez[i],
                contexts.clone(),
                buf,
                &mut offset,
            )
            .await?,
        ));
    }
    Ok(mrx)
}

#[pyfunction]
#[pyo3(signature = (path, urls, name, tensor, rate_limit_bytes_per_sec=None, max_concurrent_downloads=None, collect_layout=false))]
pub fn load_tensor_no_resharding<'py>(
    py: Python<'py>,
    path: Bound<'py, PyString>,
    urls: Bound<'py, PyString>,
    name: Bound<'py, PyString>,
    mut tensor: PyReadwriteArray1<'py, u8>,
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    collect_layout: bool,
) -> PyResult<Py<PyTuple>> {
    use crate::copy_port_client::{
        ShardOwnership, classify_shard_ownership, peer_send_max_pieces, replicated_send_ranges,
        restore_piece_order, shuffle_sharded_schedule,
    };

    let runtime = runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let path = path.to_string();
    let prefix: Vec<_> = path.split('/').rev().take(3).collect();
    let prefix = format!("{}/{}/", prefix[2], prefix[1]);
    let urls_string = urls.to_string();
    let (mut channels, mut entries) = py
        .detach(|| {
            runtime.block_on(get_channels_list_entries(
                urls_string.clone(),
                prefix.clone(),
            ))
        })
        .map_err(|s| PyOSError::new_err(format!("gRPC error: {}", s.message())))?;

    {
        let mut keep = entries.iter().map(|e| !e.is_empty());
        channels.retain(|_| keep.next().unwrap_or(false));
        entries.retain(|e| !e.is_empty());
    }
    if channels.is_empty() {
        return Err(PyOSError::new_err(format!(
            "no source has a listing for prefix {prefix}"
        )));
    }

    let tensor_name = name.to_string();
    let (shard_size, ownership) = classify_shard_ownership(&tensor_name, &entries)
        .map_err(|e| PyOSError::new_err(e.to_string()))?;
    if matches!(ownership, ShardOwnership::Replicated { .. }) {
        py.detach(|| {
            runtime.block_on(expand_replicated_channels(
                &urls_string,
                &prefix,
                &mut channels,
                peer_conns_per_source(),
            ))
        });
    }
    let name = format!("{tensor_name}/c/");
    let total_pieces = match &ownership {
        ShardOwnership::Sharded { owner, .. } => owner.len(),
        ShardOwnership::Replicated { total_pieces } => *total_pieces,
    };
    let tensor_slice = tensor.as_slice_mut()?;
    let tensor_size = tensor_slice.len();
    if shard_size * total_pieces != tensor_size {
        return Err(PyOSError::new_err(format!(
            "tensor size {tensor_size} must equal shard size {shard_size} times shard count {total_pieces}"
        )));
    }

    let ranges: Vec<(usize, usize, usize)> = match &ownership {
        ShardOwnership::Sharded {
            owner,
            pieces_per_rank,
        } => {
            let step = *pieces_per_rank;
            let mut ranges = Vec::with_capacity(channels.len());
            for i in 0..channels.len() {
                let shard_idx = i * step;
                let key = format!("{name}{shard_idx}/0");
                let Some(&idx) = owner.get(&key) else {
                    return Err(PyValueError::new_err(format!("{key} not found")));
                };
                ranges.push((idx, shard_idx, shard_idx + step));
            }
            ranges
        }
        ShardOwnership::Replicated { total_pieces } => replicated_send_ranges(
            *total_pieces,
            channels.len(),
            peer_send_max_pieces(shard_size),
        ),
    };

    let schedule = match &ownership {
        ShardOwnership::Sharded { .. } => shuffle_sharded_schedule(ranges.len(), &tensor_name),
        ShardOwnership::Replicated { .. } => (0..ranges.len()).collect(),
    };
    if matches!(ownership, ShardOwnership::Sharded { .. }) && ranges.len() > 1 {
        log::info!(
            "copy_port: shard schedule shuffled name={tensor_name} n={} first={:?}",
            schedule.len(),
            &schedule[..schedule.len().min(8)]
        );
    }

    #[cfg(target_os = "linux")]
    let (contexts, devicez, mrx) = {
        let contexts = Arc::new(if matches!(ownership, ShardOwnership::Sharded { .. }) {
            make_ib_contexts()
                .map_err(|e| PyOSError::new_err(format!("RDMA context creation error: {}", e)))?
        } else {
            Vec::new()
        });
        let devicez = Arc::new(
            (0..ranges.len())
                .map(|i| {
                    if contexts.is_empty() {
                        Vec::new()
                    } else {
                        vec![i * contexts.len() / channels.len()]
                    }
                })
                .collect::<Vec<_>>(),
        );
        let mrx = if contexts.is_empty() {
            (0..ranges.len()).map(|_| Arc::new(Vec::new())).collect()
        } else {
            let step = ranges.first().map(|(_, a, b)| b - a).unwrap_or(0);
            let mut futures = Vec::<BoxFuture<_>>::new();
            let buf: &'static mut [u8] = unsafe { std::mem::transmute(&mut tensor_slice[..]) };
            futures.push(Box::pin(register_tensor_for_download(
                shard_size,
                channels.len(),
                step,
                contexts.clone(),
                devicez.clone(),
                buf,
            )));
            py.detach(|| block_on(&runtime, futures, None))
                .ok_or_else(|| PyOSError::new_err("copy_port timed out during MR registration"))?
                .pop()
                .unwrap()
                .map_err(|s| {
                    PyOSError::new_err(format!("MR registration error: {}", s.message()))
                })?
        };
        (contexts, devicez, mrx)
    };

    let expected: Vec<usize> = ranges
        .iter()
        .map(|&(_, a, b)| (b - a) * shard_size)
        .collect();
    let mut futures = Vec::<BoxFuture<_>>::with_capacity(ranges.len());
    for &i in &schedule {
        let (idx, a, b) = ranges[i];
        let n = b - a;
        let slice = &mut tensor_slice[a * shard_size..b * shard_size];
        let slice: &'static mut [u8] =
            unsafe { mem::transmute(slice::from_raw_parts_mut(slice.as_mut_ptr(), slice.len())) };
        futures.push(Box::pin(send_entries(
            channels[idx].clone(),
            (a..b)
                .map(|j| format!("{prefix}{name}{j}/0").into_bytes())
                .collect(),
            vec![0; n],
            vec![shard_size; n],
            slice,
            format!("rank={i}"),
            #[cfg(target_os = "linux")]
            (devicez[i].clone(), contexts.clone(), mrx[i].clone()),
        )));
    }

    let mut checksum = 1;
    let num_futures = futures.len();
    let results = py
        .detach(|| match rate_limit_bytes_per_sec {
            Some(limit) if limit > 0 => {
                let concurrent = max_concurrent_downloads.unwrap_or(num_futures / 2).max(1);
                block_on_rate_limited(&runtime, futures, limit, concurrent)
            }
            _ => block_on(&runtime, futures, None),
        })
        .ok_or_else(|| PyOSError::new_err("copy_port timed out waiting for tensor shards"))?;
    let results = restore_piece_order(results, &schedule).ok_or_else(|| {
        PyOSError::new_err("copy_port shuffled download result count/order mismatch")
    })?;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
    for ((sent, sum), want) in results.into_iter().zip(expected) {
        if sent == TRANSFER_FAILED_SENTINEL {
            return Err(PyOSError::new_err(
                "copy_port gRPC/RDMA transfer failed (check Rust logs for details)",
            ));
        }
        if sent != want {
            return Err(PyOSError::new_err(format!(
                "copy_port bad sent size: wanted {want}, got {sent}"
            )));
        }
        adler32_combine(&mut checksum, sum, sent);
    }

    if collect_layout {
        return Ok((checksum, shard_size as u64, total_pieces as u64)
            .into_pyobject(py)?
            .into());
    }
    Ok((checksum,).into_pyobject(py)?.into())
}

pub fn load_tensor_into(
    path: &str,
    urls: &str,
    shard_sources: &[(String, String, usize, usize)],
    tensor: &mut [u8],
    row_size: usize,
    num_row_segments: usize,
) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    let oflags = OFlag::O_RDONLY | OFlag::O_DIRECT;
    #[cfg(not(target_os = "linux"))]
    let oflags = OFlag::O_RDONLY;

    let tensor_size = tensor.len();
    if num_row_segments == 0 || !tensor_size.is_multiple_of(num_row_segments) {
        return Err(format!(
            "tensor size {tensor_size} must be divisible by num_row_segments {num_row_segments}"
        ));
    }
    let row_segment_size = tensor_size / num_row_segments;
    if row_size == 0 || !row_segment_size.is_multiple_of(row_size) {
        return Err(format!(
            "row_segment_size {row_segment_size} must be divisible by row_size {row_size}"
        ));
    }
    if !shard_sources.len().is_multiple_of(num_row_segments) {
        return Err(format!(
            "shard sources size {} must be divisible by num_row_segments {num_row_segments}",
            shard_sources.len()
        ));
    }
    let num_shards = shard_sources.len() / num_row_segments;
    if !PAGE_SIZE.is_multiple_of(row_size) {
        return Err(format!(
            "page size {PAGE_SIZE} must be divisible by row_size {row_size}"
        ));
    }
    if num_shards == 0 || !row_size.is_multiple_of(num_shards) {
        return Err(format!(
            "row_size {row_size} must be divisible by num_shards {num_shards}"
        ));
    }
    let w_s = row_size / num_shards;
    if w_s < WIDTH && !WIDTH.is_multiple_of(w_s) {
        return Err(format!(
            "{WIDTH} must be divisible by (row_size / num_shards) {w_s}"
        ));
    }
    let delta_shards = cmp::max(1, WIDTH / w_s);
    if !num_shards.is_multiple_of(delta_shards) {
        return Err(format!(
            "num_shards {num_shards} must be divisible by delta_shards {delta_shards}"
        ));
    }
    let num_blocks = num_shards / delta_shards;
    let shard_size = row_segment_size / num_shards;

    let s: Vec<(Result<usize, OwnedFd>, usize, String)> = Vec::with_capacity(shard_sources.len());
    let mut shards = scopeguard::guard(s, |s| {
        for (idx_or_fd, _, _) in s {
            if let Err(fd) = idx_or_fd {
                let _ = close(fd);
            }
        }
    });

    let runtime = runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;

    let prefix: Vec<_> = path.split('/').rev().take(3).collect();
    if prefix.len() < 3 {
        return Err(format!("checkpoint path too short: {path}"));
    }
    let prefix = format!("{}/{}/", prefix[2], prefix[1]);
    let (channels, entries) = runtime
        .block_on(get_channels_list_entries(urls.to_string(), prefix.clone()))
        .map_err(|s| format!("gRPC error: {}", s.message()))?;

    let mut channel_indexes = HashMap::new();
    for (channel_idx, inner) in entries.into_iter().enumerate() {
        for (key, value) in inner {
            channel_indexes.insert(key, (channel_idx, value));
        }
    }
    for (key, fname, offset, size) in shard_sources {
        if *size != shard_size {
            return Err(format!(
                "bad shard size of name {key}: wanted {shard_size}, got {size}"
            ));
        }
        let empty_fname = fname.is_empty();
        let fname = format!("{path}/{fname}");
        let idx_or_fd: Result<usize, OwnedFd> = if let Some(&(idx, sz)) = channel_indexes.get(key) {
            if sz != *size {
                return Err(format!(
                    "bad shard size of name {key}: wanted {size}, got {sz}"
                ));
            }
            Ok(idx)
        } else {
            if empty_fname {
                return Err(format!("cannot load shard {key}"));
            }
            Err(open(fname.as_str(), oflags, Mode::empty())
                .map_err(|e| format!("failed to open file {fname}: {e}"))?)
        };
        shards.push((idx_or_fd, *offset, format!("{prefix}{key}")));
    }

    let ok = AtomicBool::new(true);
    let counters = (AtomicUsize::new(0), AtomicUsize::new(0));
    for row_segment_idx in 0..num_row_segments {
        let shard_idx = row_segment_idx * num_shards;
        let chunk_size = cmp::max(
            READ_SIZE * num_shards,
            row_segment_size / num_shards / PAGE_SIZE / CONCURRENCY_OUTER * num_shards * PAGE_SIZE,
        );
        let row_segment_slice = &mut tensor
            [row_segment_size * row_segment_idx..row_segment_size * (row_segment_idx + 1)];
        let base_ptr = row_segment_slice.as_ptr() as usize;
        row_segment_slice
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                let file_offset = (chunk.as_ptr() as usize - base_ptr) / num_shards;
                let min_count = chunk.len() / num_shards;
                let buf_size = min_count.div_ceil(PAGE_SIZE) * PAGE_SIZE;

                let mut vecs = vec![vec![0; buf_size + 2 * PAGE_SIZE]; delta_shards];
                let mut bufs = Vec::with_capacity(delta_shards);
                for v in vecs.iter_mut() {
                    let _ = madvise_hugepage_internal(v);
                    let o = (!(v.as_ptr() as usize) + 1) & (PAGE_SIZE - 1);
                    bufs.push(&mut v[o..o + buf_size + PAGE_SIZE]);
                }
                let mut pads = vec![0; delta_shards];
                let mut counts = (0usize, 0usize);

                for block_idx in 0..num_blocks {
                    let mut futures = Vec::<BoxFuture<_>>::with_capacity(delta_shards);

                    for idx in 0..delta_shards {
                        let shard = &shards[shard_idx + block_idx * delta_shards + idx];
                        let offset = shard.1 + file_offset;
                        pads[idx] = offset & (PAGE_SIZE - 1);
                        let o = (offset - pads[idx]) as i64;
                        match &shard.0 {
                            Ok(channel_idx) => {
                                let b = bufs[idx].as_mut_ptr();
                                let b: &'static mut [u8] = unsafe {
                                    mem::transmute(slice::from_raw_parts_mut(
                                        b.add(pads[idx]),
                                        min_count,
                                    ))
                                };
                                futures.push(Box::pin(send_entries(
                                    channels[*channel_idx].clone(),
                                    vec![shard.2.as_bytes().to_vec()],
                                    vec![file_offset],
                                    vec![min_count],
                                    b,
                                    format!("shard={shard_idx}"),
                                    #[cfg(target_os = "linux")]
                                    (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
                                )));
                            }
                            Err(fd) => {
                                let count = pread(fd.as_fd(), bufs[idx], o).unwrap_or(0);
                                if count < min_count + pads[idx] {
                                    ok.store(false, Ordering::Relaxed);
                                }
                                counts.0 += count;
                            }
                        }
                    }

                    if let Some(results) = block_on(&runtime, futures, None) {
                        for (count, _) in results {
                            if count != min_count {
                                ok.store(false, Ordering::Relaxed);
                            }
                            counts.1 += count;
                        }
                    } else {
                        ok.store(false, Ordering::Relaxed);
                    }

                    let y0 = block_idx * delta_shards * w_s;
                    let slice_size = reassembly_slice_size(chunk.len(), row_size);
                    let base_ptr = chunk.as_ptr() as usize;
                    chunk.par_chunks_mut(slice_size).for_each(|slice| {
                        let row_idx_base = (slice.as_ptr() as usize - base_ptr) / row_size;
                        for row_idx in 0..slice.len() / row_size {
                            let x1 = (row_idx_base + row_idx) * w_s;
                            let y1 = y0 + row_idx * row_size;
                            for idx in 0..delta_shards {
                                let x2 = x1 + pads[idx];
                                let y2 = y1 + idx * w_s;
                                slice[y2..y2 + w_s].copy_from_slice(&bufs[idx][x2..x2 + w_s]);
                            }
                        }
                    });
                }
                counters.0.fetch_add(counts.0, Ordering::Relaxed);
                counters.1.fetch_add(counts.1, Ordering::Relaxed);
            });
    }

    log::info!(
        "load_tensor stats: {} via file system, {} via gRPC, {} total",
        counters.0.load(Ordering::Relaxed),
        counters.1.load(Ordering::Relaxed),
        tensor_size,
    );
    if !ok.load(Ordering::Relaxed) {
        return Err("could not read files".into());
    }
    Ok(())
}

#[pyfunction]
pub fn load_tensor<'py>(
    py: Python<'py>,
    path: Bound<'py, PyString>,
    urls: Bound<'py, PyString>,
    shard_sources: Bound<'py, PyList>,
    mut tensor: PyReadwriteArray1<'py, u8>,
    row_size: usize,
    num_row_segments: usize,
) -> PyResult<()> {
    let mut parsed = Vec::with_capacity(shard_sources.len());
    for t in shard_sources.iter() {
        let err0 =
            || PyTypeError::new_err("shard_sources entry must be (name, fname, offset, size)");
        let tuple = t.downcast::<PyTuple>().map_err(|_| err0())?;
        if tuple.len() != 4 {
            return Err(err0());
        }
        let k = tuple.get_item(0)?;
        let key: String = k.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "shard_sources name {} must be a string",
                k.repr()
                    .map(|s| s.to_string())
                    .unwrap_or("<unknown>".to_string())
            ))
        })?;
        let err = || {
            PyTypeError::new_err(format!(
                "shard_sources entry must be (name, fname, offset, size) for name {key}"
            ))
        };
        let fname: String = tuple.get_item(1)?.extract().map_err(|_| err())?;
        let offset: usize = tuple.get_item(2)?.extract().map_err(|_| err())?;
        let size: usize = tuple.get_item(3)?.extract().map_err(|_| err())?;
        parsed.push((key, fname, offset, size));
    }
    let path = path.to_string();
    let urls = urls.to_string();
    let tensor_slice = tensor.as_slice_mut()?;
    py.detach(|| {
        load_tensor_into(
            &path,
            &urls,
            &parsed,
            tensor_slice,
            row_size,
            num_row_segments,
        )
        .map_err(PyOSError::new_err)
    })
}

#[cfg(test)]
mod copy_rows_tests {
    use super::*;

    struct AlignedBuf {
        ptr: *mut u8,
        len: usize,
        layout: std::alloc::Layout,
    }

    impl Drop for AlignedBuf {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.ptr, self.layout) }
        }
    }

    impl AlignedBuf {
        fn new_zeroed(len: usize) -> Self {
            assert!(len.is_multiple_of(64));
            let layout = std::alloc::Layout::from_size_align(len, 64).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, len, layout }
        }

        fn from_iter(len: usize, it: impl Iterator<Item = u8>) -> Self {
            let mut buf = Self::new_zeroed(len);
            for (dst, src) in buf.as_mut_slice().iter_mut().zip(it) {
                *dst = src;
            }
            buf
        }

        fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    fn deterministic_indices(num_rows: usize, count: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(count);
        let mut x: u64 = 0x9e3779b97f4a7c15;
        out.push(0);
        out.push((num_rows - 1) as u32);
        while out.len() < count {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push((x >> 33) as u32 % num_rows as u32);
        }
        out
    }

    #[test]
    fn copy_rows_matches_generic_for_all_row_sizes() {
        for row_size in [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072] {
            let num_rows = 257;
            let table = AlignedBuf::from_iter(
                num_rows * row_size,
                (0..num_rows * row_size).map(|i| (i % 251) as u8),
            );
            let indices = deterministic_indices(num_rows, 64);

            let mut got = AlignedBuf::new_zeroed(indices.len() * row_size);
            let mut want = AlignedBuf::new_zeroed(indices.len() * row_size);
            copy_rows(table.as_slice(), &indices, got.as_mut_slice(), row_size);
            copy_rows_generic(table.as_slice(), &indices, want.as_mut_slice(), row_size);
            assert_eq!(
                got.as_slice(),
                want.as_slice(),
                "copy_rows disagrees with generic at row_size={row_size}"
            );
        }
    }

    #[test]
    fn gather_accepts_cache_line_multiples_and_rejects_others() {
        let row_size = 768;
        let num_rows = 64;
        let table = AlignedBuf::from_iter(
            num_rows * row_size,
            (0..num_rows * row_size).map(|i| (i % 253) as u8),
        );
        let indices: Vec<u32> = (0..32).collect();
        let mut out = AlignedBuf::new_zeroed(indices.len() * row_size);

        embedding_gather_into(
            table.as_slice(),
            &indices,
            &mut [out.as_mut_slice()],
            row_size,
            2,
        )
        .expect("row_size 768 must be accepted");
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(
                &out.as_slice()[i * row_size..(i + 1) * row_size],
                &table.as_slice()[idx as usize * row_size..(idx as usize + 1) * row_size],
            );
        }

        for bad in [0usize, 32, 96, 700] {
            let err = embedding_gather_into(table.as_slice(), &indices, &mut [], bad, 2)
                .expect_err("non-multiple-of-64 row_size must be rejected");
            assert!(err.0.contains("multiple of 64"), "{}", err.0);
        }
    }

    #[test]
    #[ignore]
    fn bench_copy_rows_row_sizes() {
        for row_size in [256, 384, 512, 768, 1024, 1536, 2048, 3072] {
            let num_rows = (1usize << 30) / row_size;
            let table = AlignedBuf::new_zeroed(num_rows * row_size);
            let indices = deterministic_indices(num_rows, 1 << 20);
            let mut out = AlignedBuf::new_zeroed(indices.len() * row_size);

            let start = std::time::Instant::now();
            let iters = 8;
            for _ in 0..iters {
                copy_rows(table.as_slice(), &indices, out.as_mut_slice(), row_size);
            }
            let secs = start.elapsed().as_secs_f64();
            let gib = (iters * indices.len() * row_size) as f64 / (1u64 << 30) as f64;
            println!("row_size {row_size:5}: {:.2} GiB/s", gib / secs);
        }
    }
}

#[cfg(test)]
mod channel_config_tests {
    use super::*;

    #[test]
    fn window_mib_parses_defaults_and_disables() {
        assert_eq!(parse_window_mib(None, 64), Some(64 << 20));
        assert_eq!(parse_window_mib(Some("bogus"), 16), Some(16 << 20));
        assert_eq!(parse_window_mib(Some("8"), 64), Some(8 << 20));
        assert_eq!(parse_window_mib(Some("0"), 64), None);
        assert_eq!(parse_window_mib(Some("4096"), 64), Some(1024 << 20));
    }

    #[test]
    fn duplicated_urls_repeats_each_source() {
        assert_eq!(
            duplicated_urls("http://a:1,http://b:2", 2),
            "http://a:1,http://b:2,http://a:1,http://b:2"
        );
        assert_eq!(duplicated_urls(" http://a:1 ,, ", 1), "http://a:1");
        assert_eq!(duplicated_urls("http://a:1", 0), "");
    }

    #[test]
    fn conns_per_source_parses_and_clamps() {
        assert_eq!(parse_conns_per_source(None), 1);
        assert_eq!(parse_conns_per_source(Some("bogus")), 1);
        assert_eq!(parse_conns_per_source(Some("")), 1);
        assert_eq!(parse_conns_per_source(Some("-1")), 1);
        assert_eq!(parse_conns_per_source(Some("18446744073709551616")), 1);
        assert_eq!(parse_conns_per_source(Some("4")), 4);
        assert_eq!(parse_conns_per_source(Some("0")), 1);
        assert_eq!(parse_conns_per_source(Some("64")), 16);
    }
}

#[cfg(test)]
mod load_tensor_into_tests {
    use super::*;

    #[test]
    fn reassembly_slices_are_whole_rows_and_never_empty() {
        let row_size = 2048;
        for rows in [
            1,
            CONCURRENCY_INNER - 1,
            CONCURRENCY_INNER,
            1000,
            READ_SIZE / row_size,
        ] {
            let slice_size = reassembly_slice_size(rows * row_size, row_size);
            assert!(slice_size >= row_size, "{rows} rows: slice {slice_size}");
            assert!(
                slice_size.is_multiple_of(row_size),
                "{rows} rows: slice {slice_size}"
            );
        }
        assert_eq!(
            reassembly_slice_size(READ_SIZE, row_size),
            READ_SIZE / CONCURRENCY_INNER
        );
    }

    #[cfg(all(target_os = "linux", feature = "rdma-tests"))]
    fn supports_o_direct(dir: &std::path::Path) -> bool {
        let probe = dir.join("o_direct_probe");
        std::fs::write(&probe, [0u8; 4096]).is_ok()
            && open(
                probe.as_path(),
                OFlag::O_RDONLY | OFlag::O_DIRECT,
                Mode::empty(),
            )
            .is_ok()
    }

    #[cfg(all(target_os = "linux", feature = "rdma-tests"))]
    #[test]
    fn row_segment_with_short_tail_chunk_loads() {
        let dir = tempfile::tempdir().unwrap();
        if !supports_o_direct(dir.path()) {
            eprintln!(
                "skipping: {} does not support O_DIRECT",
                dir.path().display()
            );
            return;
        }
        let row_size = 2048;
        let tensor_size = READ_SIZE + (CONCURRENCY_INNER - 1) * row_size;
        let ckpt = dir.path().join("run/elapsed_samples_1/orbax-ckpt");
        std::fs::create_dir_all(&ckpt).unwrap();
        let data: Vec<u8> = (0..tensor_size).map(|i| (i % 251) as u8).collect();
        std::fs::write(ckpt.join("shard"), &data).unwrap();
        let shard_sources = [(
            "emb_table/c/0/0".to_string(),
            "shard".to_string(),
            0,
            tensor_size,
        )];

        let mut tensor = vec![0u8; tensor_size];
        load_tensor_into(
            ckpt.to_str().unwrap(),
            "",
            &shard_sources,
            &mut tensor,
            row_size,
            1,
        )
        .unwrap();
        assert!(tensor == data, "loaded tensor differs from the shard file");
    }
}
