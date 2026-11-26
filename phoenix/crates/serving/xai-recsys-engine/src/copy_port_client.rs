// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::cmp;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::{Duration, Instant};

use futures::future::{BoxFuture, join_all};
use lazy_static::lazy_static;
use rand::prelude::*;
use tonic::transport::{Channel, Endpoint};

use crate::adler32::adler32_combine;
use crate::emb_table::{
    duplicated_urls, get_channels, get_channels_with_endpoints, list_entries, send_entries,
    send_entries_reporting,
};
use crate::grpc_util::TRANSFER_FAILED_SENTINEL;

pub struct TensorBuf<'a> {
    pub key: String,
    pub buf: &'a mut [u8],
}

#[derive(Debug, Clone)]
pub struct DownloadMeta {
    pub prefix: String,
    pub checksums_json: String,
    pub created_timestamp: f64,
    pub entries: Vec<DenseEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseEntry {
    pub key: String,
    pub name: String,
    pub adler32: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardLayout {
    pub piece_bytes: usize,
    pub pieces: usize,
}

pub trait DownloadProgress: Send + Sync {
    fn layout(&self, layout: ShardLayout);
    fn written(&self, start: usize, end: usize);
}

#[derive(Clone)]
pub(crate) struct ProgressAt {
    pub(crate) progress: Arc<dyn DownloadProgress>,
    pub(crate) offset: usize,
}

impl ProgressAt {
    fn at(progress: &Option<Arc<dyn DownloadProgress>>, offset: usize) -> Option<ProgressAt> {
        progress.as_ref().map(|progress| ProgressAt {
            progress: progress.clone(),
            offset,
        })
    }
}

#[derive(Debug)]
pub enum CopyPortError {
    NoNewer {
        new_prefix: String,
        prev_prefix: String,
    },
    Grpc(String),
    SizeMismatch {
        key: String,
        listed: usize,
        requested: usize,
    },
    NotFound {
        key: String,
    },
    TransferFailed(String),
    Timeout(String),
    Other(String),
}

impl std::fmt::Display for CopyPortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoNewer {
                new_prefix,
                prev_prefix,
            } => write!(
                f,
                "new prefix {new_prefix} is no greater than old prefix {prev_prefix}"
            ),
            Self::Grpc(s) => write!(f, "gRPC error: {s}"),
            Self::SizeMismatch {
                key,
                listed,
                requested,
            } => write!(
                f,
                "size mismatch on {key}: listed {listed}, requested {requested}"
            ),
            Self::NotFound { key } => write!(f, "cannot find {key}"),
            Self::TransferFailed(s) => write!(f, "transfer failed: {s}"),
            Self::Timeout(s) => write!(f, "timeout: {s}"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for CopyPortError {}

const MAX_PACING_SLEEP: Duration = Duration::from_secs(60);

pub async fn resolve_copy_urls(urls: &str) -> Result<String, CopyPortError> {
    let mut out: Vec<String> = Vec::new();
    for part in urls.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (scheme, host, port) = parse_host_port(part)?;
        let lookup = format!("{host}:{port}");
        let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&lookup)
            .await
            .map_err(|e| {
                CopyPortError::Other(format!("copy_port DNS lookup failed for {lookup}: {e}"))
            })?
            .collect();
        if addrs.is_empty() {
            return Err(CopyPortError::Other(format!(
                "copy_port DNS returned no addresses for {lookup}"
            )));
        }
        let v4: Vec<_> = addrs.iter().filter(|a| a.ip().is_ipv4()).copied().collect();
        let use_addrs = if v4.is_empty() { &addrs[..] } else { &v4[..] };
        for sa in use_addrs {
            out.push(format!("{scheme}://{}:{port}", sa.ip()));
        }
    }
    if out.is_empty() {
        return Err(CopyPortError::Other(
            "copy_port urls empty after resolve".into(),
        ));
    }
    let mut seen = HashSet::new();
    out.retain(|u| seen.insert(u.clone()));
    log::info!(
        "copy_port: resolved {} endpoint(s) from input urls",
        out.len()
    );
    Ok(out.join(","))
}

fn parse_host_port(s: &str) -> Result<(&'static str, String, u16), CopyPortError> {
    let s = s.trim();
    let (scheme, s) = if let Some(rest) = s.strip_prefix("https://") {
        ("https", rest)
    } else {
        ("http", s.trim_start_matches("http://"))
    };
    let (host, port_str) = s.rsplit_once(':').ok_or_else(|| {
        CopyPortError::Other(format!(
            "bad copy_url '{s}' (want host:port or http(s)://host:port)"
        ))
    })?;
    let port: u16 = port_str
        .parse()
        .map_err(|_| CopyPortError::Other(format!("bad copy_url port in '{s}'")))?;
    if host.is_empty() {
        return Err(CopyPortError::Other(format!("bad copy_url host in '{s}'")));
    }
    Ok((scheme, host.to_string(), port))
}

fn checkpoint_prefix_from_path(path: &str) -> Result<String, CopyPortError> {
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() < 2 {
        return Err(CopyPortError::Other(format!(
            "bad path for prefix extraction (need elapsed_samples_*/run_id): {path}"
        )));
    }
    let n = parts.len();
    Ok(format!("{}/{}/", parts[n - 2], parts[n - 1]))
}

pub fn parse_elapsed_from_prefix(prefix: &str) -> Option<u64> {
    let first = prefix.split('/').next()?;
    let n = first.strip_prefix("elapsed_samples_")?;
    n.parse().ok()
}

fn newest_full_prefix(entries: &[Vec<(String, usize)>]) -> String {
    full_prefixes_newest_first(entries)
        .into_iter()
        .next()
        .unwrap_or_default()
}

fn full_prefixes_newest_first(entries: &[Vec<(String, usize)>]) -> Vec<String> {
    let n_active = entries.iter().filter(|e| !e.is_empty()).count();
    if n_active == 0 {
        return Vec::new();
    }
    let mut counts = BTreeMap::<&str, usize>::new();
    for entry_list in entries {
        if entry_list.is_empty() {
            continue;
        }
        let mut prev_prefix = "";
        for (name, _) in entry_list {
            if let Some(i) = name.find('/')
                && let Some(j) = name[i + 1..].find('/')
            {
                let prefix = &name[..i + j + 1];
                if prev_prefix != prefix {
                    prev_prefix = prefix;
                    *counts.entry(prefix).or_insert(0) += 1;
                }
            }
        }
    }
    counts
        .into_iter()
        .rev()
        .filter(|(_, count)| *count == n_active)
        .map(|(name, _)| name.to_string())
        .collect()
}

fn is_newer_prefix(prefix: &str, elapsed_samples: u64) -> bool {
    let prev = format!("elapsed_samples_{elapsed_samples:018}/");
    !prefix.is_empty() && prefix.len() >= prev.len() && prefix[..prev.len()] > prev[..]
}

fn set_or_check(slot: &mut usize, value: usize, what: &str) -> Result<(), CopyPortError> {
    if value == 0 {
        return Err(CopyPortError::Other(format!("unexpected zero {what}")));
    }
    if *slot == 0 {
        *slot = value;
    } else if *slot != value {
        return Err(CopyPortError::Other(format!(
            "{what} mismatch: expected {slot}, got {value}"
        )));
    }
    Ok(())
}

async fn connect_and_list(
    urls: &str,
    list_prefix: String,
) -> Result<(Vec<Channel>, Vec<Vec<(String, usize)>>, Vec<Endpoint>), CopyPortError> {
    let urls = resolve_copy_urls(urls).await?;
    let (channels, endpoints): (Vec<_>, Vec<_>) = get_channels_with_endpoints(urls)
        .await
        .map_err(|s| CopyPortError::Grpc(s.message().to_string()))?
        .into_iter()
        .unzip();
    let entries = list_entries(&channels, &list_prefix)
        .await
        .map_err(|s| CopyPortError::Grpc(s.message().to_string()))?;
    let (sources, entries): (Vec<_>, Vec<_>) = channels
        .into_iter()
        .zip(endpoints)
        .zip(entries)
        .filter(|(_, e)| !e.is_empty())
        .unzip();
    let (channels, endpoints): (Vec<_>, Vec<_>) = sources.into_iter().unzip();
    if channels.is_empty() {
        return Err(CopyPortError::Grpc(
            "no copy_port channels connected".into(),
        ));
    }
    log::info!(
        "copy_port: {} channel(s) with non-empty listings",
        channels.len()
    );
    Ok((channels, entries, endpoints))
}

type TransferFuture = BoxFuture<'static, (usize, u32)>;
type DenseDownloadPlan = (Vec<TransferFuture>, Vec<usize>, Vec<u8>);

type IndexedTransfer = Result<(usize, (usize, u32)), CopyPortError>;

struct TransferSlot(Mutex<Option<TransferFuture>>);

impl TransferSlot {
    fn poll(&self, cx: &mut std::task::Context<'_>) -> Poll<Result<(usize, u32), CopyPortError>> {
        let mut future = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let result = catch_unwind(AssertUnwindSafe(|| {
            future.as_mut().map(|future| future.as_mut().poll(cx))
        }));
        if matches!(result, Ok(Some(Poll::Pending))) {
            return Poll::Pending;
        }
        let dropped = catch_unwind(AssertUnwindSafe(|| drop(future.take())));
        Poll::Ready(match (result, dropped) {
            (Ok(Some(Poll::Ready(result))), Ok(())) => Ok(result),
            (Ok(None), Ok(())) => Err(CopyPortError::Other("copy_port download cancelled".into())),
            _ => Err(CopyPortError::Other(
                "copy_port download task panicked".into(),
            )),
        })
    }
}

#[derive(Default)]
struct ActiveTransfers {
    slots: Vec<Arc<TransferSlot>>,
    tasks: tokio::task::JoinSet<IndexedTransfer>,
}

impl ActiveTransfers {
    fn push(&mut self, index: usize, future: TransferFuture) {
        let slot = Arc::new(TransferSlot(Mutex::new(Some(future))));
        self.slots.push(slot.clone());
        self.tasks.spawn(std::future::poll_fn(move |cx| {
            slot.poll(cx)
                .map(|result| result.map(|result| (index, result)))
        }));
    }
}

impl Drop for ActiveTransfers {
    fn drop(&mut self) {
        let mut panicked = false;
        for slot in &self.slots {
            let mut future = slot.0.lock().unwrap_or_else(|error| error.into_inner());
            panicked |= catch_unwind(AssertUnwindSafe(|| drop(future.take()))).is_err();
        }
        self.tasks.abort_all();
        if panicked {
            log::error!(
                "copy_port: transfer destructor panicked; all active futures were cleaned up"
            );
        }
    }
}

async fn run_downloads(
    futures: Vec<TransferFuture>,
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    let limit = rate_limit_bytes_per_sec.unwrap_or(0);
    let max_concurrent = if limit == 0 {
        futures.len()
    } else {
        max_concurrent_downloads.unwrap_or(futures.len())
    };
    join_rate_limited(futures, limit, max_concurrent).await
}

fn pacing_sleep(total_bytes: u64, rate: u64, elapsed: Duration) -> Duration {
    if rate == 0 {
        return Duration::ZERO;
    }
    let expected =
        Duration::try_from_secs_f64(total_bytes as f64 / rate as f64).unwrap_or(Duration::MAX);
    let sleep = expected.saturating_sub(elapsed);
    if sleep > MAX_PACING_SLEEP {
        log::warn!(
            "rate_limiter: pacing sleep for {total_bytes} bytes exceeds \
             {MAX_PACING_SLEEP:?}; capping"
        );
    }
    sleep.min(MAX_PACING_SLEEP)
}

async fn join_rate_limited(
    futures: Vec<TransferFuture>,
    rate_limit_bytes_per_sec: u64,
    max_concurrent: usize,
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    if futures.is_empty() {
        return Ok(Vec::new());
    }
    let max_concurrent = max_concurrent.max(1);
    let start = tokio::time::Instant::now();
    let mut next_start = start;
    let mut total_bytes = 0u64;
    let mut results = vec![None; futures.len()];
    let mut remaining = futures.into_iter().enumerate();
    let mut active = ActiveTransfers::default();
    let mut failed = false;
    let mut join_error = None;
    for (index, future) in remaining.by_ref().take(max_concurrent) {
        active.push(index, future);
    }

    while !active.tasks.is_empty() || (!failed && remaining.len() > 0) {
        tokio::select! {
            biased;
            Some(joined) = active.tasks.join_next(), if !active.tasks.is_empty() => {
                match joined.unwrap_or_else(|error| {
                    Err(CopyPortError::Other(format!("copy_port download task join: {error}")))
                }) {
                    Ok((index, result)) => {
                        results[index] = Some(result);
                        if result.0 == TRANSFER_FAILED_SENTINEL {
                            failed = true;
                        } else if !failed {
                            total_bytes = total_bytes.saturating_add(result.0 as u64);
                            let now = tokio::time::Instant::now();
                            next_start = now + pacing_sleep(
                                total_bytes, rate_limit_bytes_per_sec, now - start,
                            );
                        }
                    }
                    Err(error) => {
                        failed = true;
                        join_error.get_or_insert(error);
                    }
                }
            }
            _ = tokio::time::sleep_until(next_start),
                if !failed && remaining.len() > 0 && active.tasks.len() < max_concurrent =>
            {
                let (index, future) = remaining.next().expect("queued transfer");
                active.push(index, future);
            }
        }
    }
    if let Some(error) = join_error {
        return Err(error);
    }
    if failed {
        log::error!(
            "rate_limiter: transfer failed; drained active downloads and skipped queued work"
        );
    } else {
        tokio::time::sleep_until(next_start).await;
    }
    Ok(results.into_iter().flatten().collect())
}

fn check_transfer_results(
    results: &[(usize, u32)],
    expected_sizes: &[usize],
) -> Result<(), CopyPortError> {
    for ((count, _), size) in results.iter().zip(expected_sizes.iter()) {
        if *count == TRANSFER_FAILED_SENTINEL {
            return Err(CopyPortError::TransferFailed(
                "gRPC/RDMA transfer failed (check Rust logs)".into(),
            ));
        }
        if *count != *size {
            return Err(CopyPortError::TransferFailed(format!(
                "size mismatch: wanted {size}, got {count}"
            )));
        }
    }
    Ok(())
}

fn sfence_after_download() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

pub async fn probe_newer_checkpoint(
    elapsed_samples: u64,
    urls: &str,
) -> Result<Option<(String, u64)>, CopyPortError> {
    let (_channels, entries, _) = connect_and_list(urls, String::new()).await?;
    let prefix = newest_full_prefix(&entries);
    if !is_newer_prefix(&prefix, elapsed_samples) {
        return Ok(None);
    }
    let elapsed = parse_elapsed_from_prefix(&prefix).unwrap_or(elapsed_samples);
    Ok(Some((prefix, elapsed)))
}

fn choose_prefix(
    target_prefix: Option<&str>,
    entries: &[Vec<(String, usize)>],
) -> Result<String, CopyPortError> {
    let Some(target) = target_prefix.map(|t| t.trim_end_matches('/')) else {
        return Ok(newest_full_prefix(entries));
    };
    let dir = format!("{target}/");
    for (i, inner) in entries.iter().enumerate() {
        if !inner.iter().any(|(name, _)| name.starts_with(&dir)) {
            return Err(CopyPortError::Other(format!(
                "target prefix {target} not present on channel {i}/{}",
                entries.len()
            )));
        }
    }
    Ok(target.to_string())
}

pub async fn download_dense_weight(
    elapsed_samples: u64,
    urls: &str,
    tensors: &mut [TensorBuf<'_>],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    target_prefix: Option<&str>,
) -> Result<DownloadMeta, CopyPortError> {
    let (channels, entries, _) = connect_and_list(urls, String::new()).await?;
    let prefix = choose_prefix(target_prefix, &entries)?;
    if target_prefix.is_none() && !is_newer_prefix(&prefix, elapsed_samples) {
        return Err(CopyPortError::NoNewer {
            new_prefix: prefix,
            prev_prefix: format!("elapsed_samples_{elapsed_samples:018}/"),
        });
    }

    let channel_idx = rand::rng().random_range(0..channels.len());
    let index = index_dense_listing(&prefix, &entries[channel_idx]);
    let (futures, sizes, checksums_buf) =
        build_dense_downloads(&channels[channel_idx], &index, tensors)?;

    let results =
        run_downloads(futures, rate_limit_bytes_per_sec, max_concurrent_downloads).await?;
    sfence_after_download();
    check_transfer_results(&results, &sizes)?;

    let checksums_json = String::from_utf8(checksums_buf).unwrap_or_default();
    let created_timestamp = serde_json::from_str::<serde_json::Value>(&checksums_json)
        .ok()
        .and_then(|v| v.get("created_timestamp")?.as_f64())
        .unwrap_or(0.0);

    Ok(DownloadMeta {
        prefix,
        checksums_json,
        created_timestamp,
        entries: dense_entries(&index, tensors, &results),
    })
}

pub async fn download_dense_and_embeddings(
    elapsed_samples: u64,
    urls: &str,
    tensors: &mut [TensorBuf<'_>],
    emb: &mut [u8],
    pe: Option<&mut [u8]>,
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<(DownloadMeta, u32, Option<u32>), CopyPortError> {
    let trainer_conns_per_source = crate::emb_table::trainer_conns_per_source();
    let t_all = Instant::now();
    let (channels, entries, endpoints) = connect_and_list(urls, String::new()).await?;
    let prefix = newest_full_prefix(&entries);
    if !is_newer_prefix(&prefix, elapsed_samples) {
        return Err(CopyPortError::NoNewer {
            new_prefix: prefix,
            prev_prefix: format!("elapsed_samples_{elapsed_samples:018}/"),
        });
    }

    let channel_idx = rand::rng().random_range(0..channels.len());
    let index = index_dense_listing(&prefix, &entries[channel_idx]);
    let (futures, sizes, checksums_buf) =
        build_dense_downloads(&channels[channel_idx], &index, tensors)?;
    let results =
        run_downloads(futures, rate_limit_bytes_per_sec, max_concurrent_downloads).await?;
    sfence_after_download();
    check_transfer_results(&results, &sizes)?;
    let dense = dense_entries(&index, tensors, &results);

    let checksums_json = String::from_utf8(checksums_buf).unwrap_or_default();
    let created_timestamp = serde_json::from_str::<serde_json::Value>(&checksums_json)
        .ok()
        .and_then(|v| v.get("created_timestamp")?.as_f64())
        .unwrap_or(0.0);
    let bytes: u64 = tensors.iter().map(|t| t.buf.len() as u64).sum();
    let secs = t_all.elapsed().as_secs_f64();
    let gbs = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    log::info!(
        "copy_port: dense weights loaded bytes={bytes} in {:.2}s ({:.2} GB/s)",
        secs,
        gbs
    );

    let prefix_slash = if prefix.ends_with('/') {
        prefix.clone()
    } else {
        format!("{prefix}/")
    };
    let stripped: Vec<Vec<(String, usize)>> = entries
        .iter()
        .map(|list| {
            list.iter()
                .filter_map(|(n, sz)| {
                    n.strip_prefix(&prefix_slash)
                        .or_else(|| n.strip_prefix(&prefix).and_then(|s| s.strip_prefix('/')))
                        .map(|rest| (rest.to_string(), *sz))
                })
                .collect()
        })
        .collect();

    let t_emb = Instant::now();
    let emb_ck = download_sharded_with_channels(
        &channels,
        &stripped,
        &prefix_slash,
        "emb_table",
        emb,
        rate_limit_bytes_per_sec,
        max_concurrent_downloads,
        &endpoints,
        trainer_conns_per_source,
        None,
    )
    .await?;
    let secs = t_emb.elapsed().as_secs_f64();
    let gbs = if secs > 0.0 {
        emb.len() as f64 / secs / 1e9
    } else {
        0.0
    };
    log::info!(
        "copy_port: emb_table loaded bytes={} in {:.2}s ({:.2} GB/s)",
        emb.len(),
        secs,
        gbs
    );

    let pe_ck = if let Some(pe_buf) = pe.filter(|b| !b.is_empty()) {
        log::info!(
            "copy_port: downloading post_embeddings prefix={prefix_slash} bytes={}",
            pe_buf.len()
        );
        Some(
            download_sharded_with_channels(
                &channels,
                &stripped,
                &prefix_slash,
                "post_embeddings.embeddings",
                pe_buf,
                rate_limit_bytes_per_sec,
                max_concurrent_downloads,
                &endpoints,
                trainer_conns_per_source,
                None,
            )
            .await?,
        )
    } else {
        None
    };

    Ok((
        DownloadMeta {
            prefix,
            checksums_json,
            created_timestamp,
            entries: dense,
        },
        emb_ck,
        pe_ck,
    ))
}

async fn download_sharded_with_channels(
    channels: &[Channel],
    entries: &[Vec<(String, usize)>],
    prefix: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    endpoints: &[Endpoint],
    trainer_conns_per_source: usize,
    progress: Option<Arc<dyn DownloadProgress>>,
) -> Result<u32, CopyPortError> {
    let layout = ShardedLayout::from_listing(name, entries)?;
    layout.check_buffer_size(name, buf.len())?;
    let concurrent = max_concurrent_downloads.or(Some((channels.len() / 2).max(1)));
    let mut connection_limit = trainer_conns_per_source.clamp(1, 16);
    if connection_limit > 1 && std::env::var("XAI_RECSYS_RDMA").as_deref() == Ok("1") {
        log::warn!("copy_port: trainer connections ignored while XAI_RECSYS_RDMA=1");
        connection_limit = 1;
    }
    let trainer_opt_in =
        connection_limit > 1 && matches!(layout.ownership, ShardOwnership::Sharded { .. });
    let limit = if rate_limit_bytes_per_sec.is_some_and(|r| r > 0) {
        concurrent
    } else {
        max_concurrent_downloads
    };
    let connection_limit = connection_limit.min(limit.unwrap_or(16).max(1));
    let striped = trainer_opt_in && connection_limit > 1;
    let (futures, expected, schedule) = if striped {
        spawn_striped_trainer_downloads(
            &layout,
            prefix,
            name,
            channels,
            endpoints,
            buf,
            connection_limit,
            &progress,
        )
        .await?
    } else {
        spawn_sharded_downloads(&layout, prefix, name, channels, buf, &progress).await?
    };
    let results = if trainer_opt_in
        && rate_limit_bytes_per_sec.is_none_or(|r| r == 0)
        && let Some(limit) = max_concurrent_downloads
    {
        let mut results = Vec::new();
        let mut iter = futures.into_iter();
        loop {
            let batch: Vec<_> = iter.by_ref().take(limit.max(1)).collect();
            if batch.is_empty() {
                break;
            }
            let batch = run_downloads(batch, None, None).await?;
            let failed = batch.iter().any(|r| r.0 == TRANSFER_FAILED_SENTINEL);
            results.extend(batch);
            if failed {
                break;
            }
        }
        results
    } else {
        run_downloads(futures, rate_limit_bytes_per_sec, concurrent).await?
    };
    sfence_after_download();
    let results = results_in_piece_order(results, &schedule)?;
    combine_transfer_checksums(&results, &expected)
}

pub const BUNDLE_MANIFEST_NAME: &str = "export/MANIFEST.json";

pub async fn find_bundle_checkpoint(
    elapsed_samples: u64,
    urls: &str,
) -> Result<Option<(String, u64)>, CopyPortError> {
    let (_channels, entries, _) = connect_and_list(urls, String::new()).await?;
    select_bundle_checkpoint(&entries, elapsed_samples)
}

fn select_bundle_checkpoint(
    entries: &[Vec<(String, usize)>],
    elapsed_samples: u64,
) -> Result<Option<(String, u64)>, CopyPortError> {
    let mut newest_unbundled: Option<String> = None;
    for prefix in full_prefixes_newest_first(entries) {
        if !is_newer_prefix(&prefix, elapsed_samples) {
            break;
        }
        let manifest_name = format!("{prefix}/{BUNDLE_MANIFEST_NAME}");
        let on_all_channels = entries.iter().filter(|l| !l.is_empty()).all(|list| {
            list.iter()
                .any(|(name, size)| name == &manifest_name && *size > 0)
        });
        if on_all_channels {
            if let Some(skipped) = &newest_unbundled {
                log::warn!(
                    "copy_port: newest checkpoint {skipped} has no complete \
                     {BUNDLE_MANIFEST_NAME}; using older bundled checkpoint {prefix}"
                );
            }
            let elapsed = parse_elapsed_from_prefix(&prefix).unwrap_or(elapsed_samples);
            return Ok(Some((prefix, elapsed)));
        }
        newest_unbundled.get_or_insert(manifest_name);
    }
    match newest_unbundled {
        Some(key) => Err(CopyPortError::NotFound { key }),
        None => Ok(None),
    }
}

pub async fn download_named_files(
    prefix: &str,
    urls: &str,
    names: &[String],
) -> Result<Vec<Vec<u8>>, CopyPortError> {
    let prefix = format!("{}/", prefix.trim_end_matches('/'));
    let (channels, entries, _) = connect_and_list(urls, prefix.clone()).await?;
    let complete: Vec<usize> = (0..channels.len())
        .filter(|&i| {
            let listing: HashMap<&str, usize> =
                entries[i].iter().map(|(n, s)| (n.as_str(), *s)).collect();
            names
                .iter()
                .all(|n| listing.contains_key(n.strip_prefix(&prefix).unwrap_or(n)))
        })
        .collect();
    let Some(&channel_idx) = complete.as_slice().choose(&mut rand::rng()) else {
        return Err(CopyPortError::NotFound {
            key: format!(
                "{prefix}{{{}}} (no channel lists all files)",
                names.join(",")
            ),
        });
    };
    let listing: HashMap<&str, usize> = entries[channel_idx]
        .iter()
        .map(|(name, size)| (name.as_str(), *size))
        .collect();

    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(names.len());
    let mut sizes: Vec<usize> = Vec::with_capacity(names.len());
    let mut futures: Vec<TransferFuture> = Vec::with_capacity(names.len());
    for name in names {
        let relative = name.strip_prefix(&prefix).unwrap_or(name);
        let Some(&size) = listing.get(relative) else {
            return Err(CopyPortError::NotFound {
                key: format!("{prefix}{relative}"),
            });
        };
        let full_name = format!("{prefix}{relative}");
        let mut buf = vec![0u8; size];
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut buf[..]) };
        buffers.push(buf);
        sizes.push(size);
        futures.push(Box::pin(send_entries(
            channels[channel_idx].clone(),
            vec![full_name.into_bytes()],
            vec![0],
            vec![size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        )));
    }

    let results = run_downloads(futures, None, None).await?;
    sfence_after_download();
    check_transfer_results(&results, &sizes)?;
    Ok(buffers)
}

pub async fn download_embedding_table(
    path: &str,
    urls: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
) -> Result<u32, CopyPortError> {
    download_embedding_table_with_layout(
        path,
        urls,
        name,
        buf,
        rate_limit_bytes_per_sec,
        max_concurrent_downloads,
        false,
        None,
    )
    .await
    .map(|(checksum, _)| checksum)
}

#[allow(clippy::too_many_arguments)]
pub async fn download_embedding_table_with_layout(
    path: &str,
    urls: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    replicas: bool,
    progress: Option<Arc<dyn DownloadProgress>>,
) -> Result<(u32, ShardLayout), CopyPortError> {
    download_embedding_table_with_conns(
        path,
        urls,
        name,
        buf,
        rate_limit_bytes_per_sec,
        max_concurrent_downloads,
        crate::emb_table::peer_conns_per_source(),
        crate::emb_table::trainer_conns_per_source(),
        replicas,
        progress,
    )
    .await
}

async fn download_embedding_table_with_conns(
    path: &str,
    urls: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    peer_conns_per_source: usize,
    trainer_conns_per_source: usize,
    replicas: bool,
    progress: Option<Arc<dyn DownloadProgress>>,
) -> Result<(u32, ShardLayout), CopyPortError> {
    let prefix = checkpoint_prefix_from_path(path)?;
    let resolved = resolve_copy_urls(urls).await?;
    let (mut channels, entries, endpoints) = connect_and_list(&resolved, prefix.clone()).await?;
    let mut layout = ShardedLayout::from_listing(name, &entries)?;
    if replicas {
        layout.ownership = layout.ownership.into_replicas();
    }
    layout.check_buffer_size(name, buf.len())?;
    let shard_layout = ShardLayout {
        piece_bytes: layout.piece_bytes,
        pieces: layout.total_pieces(),
    };
    if let Some(progress) = &progress {
        progress.layout(shard_layout);
    }
    if trainer_conns_per_source > 1 && matches!(layout.ownership, ShardOwnership::Sharded { .. }) {
        let checksum = download_sharded_with_channels(
            &channels,
            &entries,
            &prefix,
            name,
            buf,
            rate_limit_bytes_per_sec,
            max_concurrent_downloads,
            &endpoints,
            trainer_conns_per_source,
            progress,
        )
        .await?;
        return Ok((checksum, shard_layout));
    }
    if matches!(layout.ownership, ShardOwnership::Replicated { .. }) {
        crate::emb_table::expand_replicated_channels(
            &resolved,
            &prefix,
            &mut channels,
            peer_conns_per_source,
        )
        .await;
    }

    let (futures, expected, schedule) =
        spawn_sharded_downloads(&layout, &prefix, name, &channels, buf, &progress).await?;
    let concurrent = max_concurrent_downloads.or(Some((channels.len() / 2).max(1)));
    let results = run_downloads(futures, rate_limit_bytes_per_sec, concurrent).await?;
    sfence_after_download();
    let results = results_in_piece_order(results, &schedule)?;
    let checksum = combine_transfer_checksums(&results, &expected)?;
    Ok((checksum, shard_layout))
}

pub const PEER_PIECE_MAX_BYTES: usize = 1 << 30;

const PROGRESSIVE_LIST_INTERVAL: Duration = Duration::from_millis(200);
const PROGRESSIVE_LIST_TIMEOUT: Duration = Duration::from_secs(5);
const PROGRESSIVE_LIST_FRESH: Duration = Duration::from_secs(2);
const PROGRESSIVE_SOURCE_FAILURES: u32 = 8;
const PROGRESSIVE_PIECE_ATTEMPTS: u32 = 6;
const PROGRESSIVE_RETRY_BACKOFF: Duration = Duration::from_millis(250);
const MAX_LISTED_PIECES: usize = 1 << 20;

impl ShardLayout {
    pub fn split(self, max_piece_bytes: usize) -> ShardLayout {
        if self.piece_bytes <= max_piece_bytes {
            return self;
        }
        let fewest = self.piece_bytes.div_ceil(max_piece_bytes.max(1));
        (fewest..=fewest * 4)
            .find(|&k| {
                self.piece_bytes.is_multiple_of(k) && (self.piece_bytes / k).is_multiple_of(64)
            })
            .map_or(self, |k| ShardLayout {
                piece_bytes: self.piece_bytes / k,
                pieces: self.pieces * k,
            })
    }
}

struct ProgressiveSource {
    url: String,
    list: Channel,
    sends: Vec<Channel>,
    next_send: usize,
    piece_bytes: usize,
    listed: Vec<bool>,
    skipped: bool,
    listing: bool,
    next_list: tokio::time::Instant,
    listed_at: Option<tokio::time::Instant>,
    inflight: usize,
    failures: u32,
    retry_at: tokio::time::Instant,
}

impl ProgressiveSource {
    fn lists(&self, start: usize) -> bool {
        self.piece_bytes != 0 && self.listed.get(start / self.piece_bytes) == Some(&true)
    }

    fn can_send(&self, now: tokio::time::Instant) -> bool {
        self.listed_at
            .is_some_and(|at| now.duration_since(at) < PROGRESSIVE_LIST_FRESH)
            && self.failures < PROGRESSIVE_SOURCE_FAILURES
            && now >= self.retry_at
            && self.inflight < self.sends.len()
    }

    fn record_listing(
        &mut self,
        listing: Option<(usize, Vec<bool>)>,
        now: tokio::time::Instant,
    ) -> bool {
        self.listing = false;
        self.next_list = now + PROGRESSIVE_LIST_INTERVAL;
        let Some((piece_bytes, listed)) = listing.filter(|_| !self.skipped) else {
            return false;
        };
        self.listed_at = Some(now);
        let before = (self.piece_bytes == piece_bytes).then_some(&self.listed);
        let appeared = listed
            .iter()
            .enumerate()
            .any(|(j, &is_listed)| is_listed && before.and_then(|b| b.get(j)) != Some(&true));
        self.piece_bytes = piece_bytes;
        self.listed = listed;
        appeared
    }
}

type ProgressiveListing = (usize, Option<(usize, Vec<bool>)>);

async fn connect_progressive_sources(
    urls: &str,
    sends_per_source: usize,
) -> Result<Vec<ProgressiveSource>, CopyPortError> {
    let resolved = resolve_copy_urls(urls).await?;
    let connected = join_all(
        resolved
            .split(',')
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .map(|url| async move {
                let channels = get_channels(duplicated_urls(url, sends_per_source + 1)).await;
                (url.to_string(), channels)
            }),
    )
    .await;
    let now = tokio::time::Instant::now();
    let mut sources = Vec::new();
    for (url, channels) in connected {
        match channels {
            Ok(mut channels) if channels.len() >= 2 => {
                let list = channels.remove(0);
                sources.push(ProgressiveSource {
                    url,
                    list,
                    sends: channels,
                    next_send: 0,
                    piece_bytes: 0,
                    listed: Vec::new(),
                    skipped: false,
                    listing: false,
                    next_list: now,
                    listed_at: None,
                    inflight: 0,
                    failures: 0,
                    retry_at: now,
                });
            }
            Ok(_) => log::warn!("copy_port: progressive source {url} did not connect"),
            Err(e) => log::warn!("copy_port: progressive source {url}: {}", e.message()),
        }
    }
    if sources.is_empty() {
        return Err(CopyPortError::Grpc(
            "no copy_port channels connected".into(),
        ));
    }
    Ok(sources)
}

fn listed_pieces(name: &str, entries: &[(String, usize)]) -> (usize, Vec<bool>) {
    let piece_prefix = format!("{name}/c/");
    let mut piece_bytes = 0;
    let mut listed = Vec::new();
    for (entry, size) in entries {
        let Some(j) = entry
            .strip_prefix(&piece_prefix)
            .and_then(|rest| rest.strip_suffix("/0"))
            .and_then(|j| j.parse::<usize>().ok())
            .filter(|&j| j < MAX_LISTED_PIECES)
        else {
            continue;
        };
        if *size == 0 || (piece_bytes != 0 && *size != piece_bytes) {
            return (0, Vec::new());
        }
        piece_bytes = *size;
        if listed.len() <= j {
            listed.resize(j + 1, false);
        }
        listed[j] = true;
    }
    (piece_bytes, listed)
}

fn list_progressive_source(
    listings: &mut tokio::task::JoinSet<ProgressiveListing>,
    sources: &mut [ProgressiveSource],
    s: usize,
    prefix: &str,
    name: &str,
) {
    sources[s].listing = true;
    let list = sources[s].list.clone();
    let (prefix, name) = (prefix.to_string(), name.to_string());
    listings.spawn(async move {
        let listed = tokio::time::timeout(
            PROGRESSIVE_LIST_TIMEOUT,
            list_entries(slice::from_ref(&list), &prefix),
        )
        .await;
        let listed = match listed {
            Ok(Ok(mut entries)) => Some(listed_pieces(&name, &entries.pop().unwrap_or_default())),
            _ => None,
        };
        (s, listed)
    });
}

fn next_progressive_unit(
    sources: &[ProgressiveSource],
    claimed: &[bool],
    unit_bytes: usize,
    now: tokio::time::Instant,
) -> Option<(usize, usize)> {
    claimed
        .iter()
        .enumerate()
        .filter(|&(_, &claimed)| !claimed)
        .find_map(|(j, _)| {
            sources
                .iter()
                .enumerate()
                .filter(|(_, source)| source.can_send(now) && source.lists(j * unit_bytes))
                .min_by_key(|(_, source)| (source.failures, source.inflight))
                .map(|(s, _)| (s, j))
        })
}

#[allow(clippy::too_many_arguments)]
pub async fn download_embedding_table_progressive(
    path: &str,
    urls: &str,
    name: &str,
    buf: &mut [u8],
    rate_limit_bytes_per_sec: Option<u64>,
    max_concurrent_downloads: Option<usize>,
    progress: Option<Arc<dyn DownloadProgress>>,
    stall_timeout: Duration,
) -> Result<(u32, ShardLayout), CopyPortError> {
    let prefix = checkpoint_prefix_from_path(path)?;
    let mut sources =
        connect_progressive_sources(urls, crate::emb_table::peer_conns_per_source()).await?;
    let max_concurrent = max_concurrent_downloads
        .unwrap_or_else(|| sources.iter().map(|s| s.sends.len()).sum())
        .max(1);
    let rate = rate_limit_bytes_per_sec.unwrap_or(0);
    let send_timeout = stall_timeout / 2;
    let (base, buf_len) = (buf.as_mut_ptr() as usize, buf.len());
    let start = tokio::time::Instant::now();
    let (mut next_start, mut last_progress) = (start, start);
    let mut listings = tokio::task::JoinSet::new();
    let mut unit: Option<ShardLayout> = None;
    let (mut claimed, mut checksums, mut attempts) = (Vec::new(), Vec::new(), Vec::new());
    let mut running: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut active = ActiveTransfers::default();
    let (mut next_index, mut done, mut total_bytes) = (0usize, 0usize, 0u64);
    let (mut starved_since, mut starved) = (None, Duration::ZERO);

    while unit.is_none_or(|unit| done < unit.pieces) {
        let now = tokio::time::Instant::now();
        for s in 0..sources.len() {
            if !sources[s].listing && now >= sources[s].next_list {
                list_progressive_source(&mut listings, &mut sources, s, &prefix, name);
            }
        }
        if let Some(layout) = unit {
            let piece = layout.piece_bytes;
            while active.tasks.len() < max_concurrent && now >= next_start {
                let Some((s, j)) = next_progressive_unit(&sources, &claimed, piece, now) else {
                    if active.tasks.is_empty() {
                        starved_since.get_or_insert(now);
                    }
                    break;
                };
                if let Some(since) = starved_since.take() {
                    starved += now - since;
                }
                let source = &mut sources[s];
                let channel = source.sends[source.next_send % source.sends.len()].clone();
                source.next_send += 1;
                source.inflight += 1;
                claimed[j] = true;
                let dest: &'static mut [u8] =
                    unsafe { slice::from_raw_parts_mut((base + j * piece) as *mut u8, piece) };
                let label = format!("source={} piece={j}", source.url);
                let send = send_entries(
                    channel,
                    vec![
                        format!("{prefix}{name}/c/{}/0", j * piece / source.piece_bytes)
                            .into_bytes(),
                    ],
                    vec![j * piece % source.piece_bytes],
                    vec![piece],
                    dest,
                    label.clone(),
                    #[cfg(target_os = "linux")]
                    (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
                );
                active.push(
                    next_index,
                    Box::pin(async move {
                        tokio::time::timeout(send_timeout, send)
                            .await
                            .unwrap_or_else(|_| {
                                log::warn!("copy_port: {label} took over {send_timeout:?}");
                                (TRANSFER_FAILED_SENTINEL, 0)
                            })
                    }),
                );
                running.insert(next_index, (s, j));
                next_index += 1;
            }
        }
        if sources
            .iter()
            .all(|s| s.failures >= PROGRESSIVE_SOURCE_FAILURES)
        {
            return Err(CopyPortError::TransferFailed(format!(
                "{name}: every progressive source failed {PROGRESSIVE_SOURCE_FAILURES} \
                 sends in a row"
            )));
        }
        if now - last_progress > stall_timeout {
            return Err(CopyPortError::Timeout(format!(
                "{name}: no new piece from {} peer(s) in {stall_timeout:?} ({done} done)",
                sources.len()
            )));
        }

        let mut wake = now + PROGRESSIVE_LIST_INTERVAL;
        for source in sources.iter().filter(|source| !source.listing) {
            wake = wake.min(source.next_list);
        }
        if next_start > now {
            wake = wake.min(next_start);
        }
        tokio::select! {
            biased;
            Some(joined) = listings.join_next(), if !listings.is_empty() => {
                let (s, listing) = joined.map_err(|error| {
                    CopyPortError::Other(format!("copy_port listing task join: {error}"))
                })?;
                let now = tokio::time::Instant::now();
                if sources[s].record_listing(listing, now) {
                    last_progress = now;
                }
                if unit.is_none()
                    && let Some(first) = sources.iter().find(|s| s.piece_bytes != 0)
                {
                    if buf_len % first.piece_bytes != 0 {
                        return Err(CopyPortError::SizeMismatch {
                            key: name.to_string(),
                            listed: first.piece_bytes,
                            requested: buf_len,
                        });
                    }
                    let layout = ShardLayout {
                        piece_bytes: first.piece_bytes,
                        pieces: buf_len / first.piece_bytes,
                    }
                    .split(PEER_PIECE_MAX_BYTES);
                    (claimed, checksums, attempts) = (
                        vec![false; layout.pieces],
                        vec![None; layout.pieces],
                        vec![0u32; layout.pieces],
                    );
                    if let Some(progress) = &progress {
                        progress.layout(layout);
                    }
                    unit = Some(layout);
                }
                if let Some(layout) = unit {
                    for source in sources.iter_mut().filter(|source| !source.skipped) {
                        let whole = layout.piece_bytes * layout.pieces;
                        if source.piece_bytes % layout.piece_bytes != 0
                            || source.piece_bytes * source.listed.len() > whole
                        {
                            log::warn!(
                                "copy_port: {name} from {} lists {} pieces of {} bytes, \
                                 not {} of {}; skipping it",
                                source.url,
                                source.listed.len(),
                                source.piece_bytes,
                                layout.pieces,
                                layout.piece_bytes
                            );
                            (source.piece_bytes, source.listed) = (0, Vec::new());
                            source.skipped = true;
                        }
                    }
                }
            }
            Some(joined) = active.tasks.join_next(), if !active.tasks.is_empty() => {
                let (index, (sent, checksum)) = joined.unwrap_or_else(|error| {
                    Err(CopyPortError::Other(format!("copy_port download task join: {error}")))
                })?;
                let (s, j) = running.remove(&index).expect("a running transfer");
                let piece = unit.expect("a layout before any transfer").piece_bytes;
                let source = &mut sources[s];
                source.inflight -= 1;
                let now = tokio::time::Instant::now();
                if sent == piece {
                    checksums[j] = Some(checksum);
                    done += 1;
                    source.failures = 0;
                    total_bytes += sent as u64;
                    last_progress = now;
                    next_start = now + pacing_sleep(total_bytes, rate, now - start);
                    if let Some(progress) = &progress {
                        sfence_after_download();
                        progress.written(j * piece, (j + 1) * piece);
                    }
                } else {
                    claimed[j] = false;
                    attempts[j] += 1;
                    source.failures += 1;
                    source.retry_at = now + PROGRESSIVE_RETRY_BACKOFF * (1 << source.failures.min(5));
                    log::warn!(
                        "copy_port: {name} piece {j} from {} failed (attempt {}, {} in a row)",
                        source.url, attempts[j], source.failures
                    );
                    if attempts[j] >= PROGRESSIVE_PIECE_ATTEMPTS {
                        return Err(CopyPortError::TransferFailed(format!(
                            "{name} piece {j} failed {} times", attempts[j]
                        )));
                    }
                }
            }
            _ = tokio::time::sleep_until(wake) => {}
        }
    }
    sfence_after_download();
    let layout = unit.expect("a finished download has a layout");
    let mut checksum = 1u32;
    for piece_checksum in checksums {
        adler32_combine(
            &mut checksum,
            piece_checksum.expect("every piece downloaded"),
            layout.piece_bytes,
        );
    }
    let secs = start.elapsed().as_secs_f64();
    log::info!(
        "copy_port: {name} progressive download of {buf_len} bytes from {} peer(s) in \
         {secs:.2}s ({:.2} GB/s), {:.1}s waiting for pieces",
        sources.len(),
        buf_len as f64 / secs.max(1e-9) / 1e9,
        starved.as_secs_f64()
    );
    Ok((checksum, layout))
}

struct DenseListing<'a> {
    names_sizes: HashMap<&'a str, (&'a str, usize)>,
    checksums_name: String,
    checksums_size: usize,
}

fn index_dense_listing<'a>(prefix: &str, entries: &'a [(String, usize)]) -> DenseListing<'a> {
    let checksums_name = format!("{prefix}/checksums.0.json");
    let mut names_sizes = HashMap::new();
    let mut checksums_size = 0usize;
    for (name, size) in entries {
        let i = prefix.len() + 1;
        if !(name.starts_with(prefix) && name.len() > i) {
            continue;
        }
        if let Some(j) = name[i..].rfind("/c") {
            names_sizes.insert(&name[i..i + j], (name.as_str(), *size));
        } else if name.as_str() == checksums_name {
            checksums_size = *size;
        }
    }
    DenseListing {
        names_sizes,
        checksums_name,
        checksums_size,
    }
}

fn build_dense_downloads(
    channel: &Channel,
    index: &DenseListing<'_>,
    tensors: &mut [TensorBuf<'_>],
) -> Result<DenseDownloadPlan, CopyPortError> {
    let mut sizes = Vec::with_capacity(tensors.len() + 1);
    let mut futures = Vec::with_capacity(tensors.len() + 1);

    for t in tensors.iter_mut() {
        let Some(&(full_name, size)) = index.names_sizes.get(t.key.as_str()) else {
            return Err(CopyPortError::NotFound { key: t.key.clone() });
        };
        if t.buf.len() != size {
            return Err(CopyPortError::SizeMismatch {
                key: t.key.clone(),
                listed: size,
                requested: t.buf.len(),
            });
        }
        sizes.push(size);
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut t.buf[..]) };
        let fut: TransferFuture = Box::pin(send_entries(
            channel.clone(),
            vec![full_name.as_bytes().to_vec()],
            vec![0],
            vec![size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        ));
        futures.push(fut);
    }

    let mut checksums = vec![0u8; index.checksums_size];
    if index.checksums_size != 0 {
        sizes.push(index.checksums_size);
        let b: &'static mut [u8] = unsafe { mem::transmute(&mut checksums[..]) };
        let fut: TransferFuture = Box::pin(send_entries(
            channel.clone(),
            vec![index.checksums_name.as_bytes().to_vec()],
            vec![0],
            vec![index.checksums_size],
            b,
            String::new(),
            #[cfg(target_os = "linux")]
            (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
        ));
        futures.push(fut);
    }

    Ok((futures, sizes, checksums))
}

fn dense_entries(
    index: &DenseListing<'_>,
    tensors: &[TensorBuf<'_>],
    results: &[(usize, u32)],
) -> Vec<DenseEntry> {
    tensors
        .iter()
        .zip(results)
        .map(|(t, &(_, adler32))| DenseEntry {
            key: t.key.clone(),
            name: index.names_sizes[t.key.as_str()].0.to_owned(),
            adler32,
        })
        .collect()
}

lazy_static! {
    pub(crate) static ref PEER_SEND_MAX_BYTES: usize = {
        std::env::var("COPY_PORT_PEER_SEND_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(16 << 30)
    };
}

pub(crate) fn replicated_send_ranges(
    total: usize,
    n_channels: usize,
    max_pieces_per_send: usize,
) -> Vec<(usize, usize, usize)> {
    if total == 0 || n_channels == 0 {
        return Vec::new();
    }
    let cap = cmp::max(1, cmp::min(max_pieces_per_send, total.div_ceil(n_channels)));
    let mut ranges = Vec::new();
    let mut lo = 0;
    while lo < total {
        let hi = cmp::min(total, lo.saturating_add(cap));
        ranges.push((ranges.len() % n_channels, lo, hi));
        lo = hi;
    }
    ranges
}

pub(crate) fn peer_send_max_pieces(piece_bytes: usize) -> usize {
    cmp::max(1, *PEER_SEND_MAX_BYTES / cmp::max(1, piece_bytes))
}

pub(crate) enum ShardOwnership {
    Sharded {
        owner: HashMap<String, usize>,
        pieces_per_rank: usize,
    },
    Replicated {
        total_pieces: usize,
    },
}

impl ShardOwnership {
    fn into_replicas(self) -> ShardOwnership {
        match self {
            ShardOwnership::Sharded {
                owner,
                pieces_per_rank,
            } if pieces_per_rank == owner.len() => ShardOwnership::Replicated {
                total_pieces: owner.len(),
            },
            other => other,
        }
    }
}

pub(crate) fn classify_shard_ownership(
    name: &str,
    entries: &[Vec<(String, usize)>],
) -> Result<(usize, ShardOwnership), CopyPortError> {
    let name_prefix = format!("{name}/c/");
    let n_channels = entries.len();
    let mut piece_bytes = 0usize;
    let mut listings: HashMap<String, (usize, usize)> = HashMap::new();
    let mut per_channel = vec![0usize; n_channels];

    for (channel_idx, inner) in entries.iter().enumerate() {
        for (key, size) in inner {
            if !key.starts_with(&name_prefix) {
                continue;
            }
            set_or_check(&mut piece_bytes, *size, "shard piece size")?;
            listings.entry(key.clone()).or_insert((channel_idx, 0)).1 += 1;
            per_channel[channel_idx] += 1;
        }
    }
    if piece_bytes == 0 || listings.is_empty() {
        return Err(CopyPortError::NotFound {
            key: name.to_string(),
        });
    }

    let total = listings.len();
    let mut pieces_per_rank = 0usize;
    let mut zero_ok = true;
    for (channel_idx, &count) in per_channel.iter().enumerate() {
        if count == 0 {
            log::warn!("copy_port: channel {channel_idx} listed 0 {name} pieces; skipping");
            continue;
        }
        if pieces_per_rank == 0 {
            pieces_per_rank = count;
        } else if count != pieces_per_rank {
            zero_ok = false;
        }
    }
    let sharded = zero_ok && pieces_per_rank != 0 && listings.values().all(|&(_, c)| c == 1);
    if sharded {
        return Ok((
            piece_bytes,
            ShardOwnership::Sharded {
                owner: listings.into_iter().map(|(k, (c, _))| (k, c)).collect(),
                pieces_per_rank,
            },
        ));
    }
    let replicated =
        listings.values().all(|&(_, c)| c == n_channels) && per_channel.iter().all(|&c| c == total);
    if replicated {
        return Ok((
            piece_bytes,
            ShardOwnership::Replicated {
                total_pieces: total,
            },
        ));
    }
    Err(CopyPortError::Other(format!(
        "inconsistent shard ownership for {name}: {total} pieces across {n_channels} channels \
         are neither exclusively sharded nor fully replicated"
    )))
}

pub(crate) fn shuffle_sharded_schedule(n: usize, name: &str) -> Vec<usize> {
    let mut schedule: Vec<usize> = (0..n).collect();
    if n <= 1 {
        return schedule;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Ok(id) = std::env::var("POD_NAME").or_else(|_| std::env::var("HOSTNAME")) {
        id.hash(&mut hasher);
    }
    name.hash(&mut hasher);
    let mut rng = rand::rngs::StdRng::seed_from_u64(hasher.finish());
    schedule.shuffle(&mut rng);
    schedule
}

pub(crate) fn restore_piece_order<T>(shuffled: Vec<T>, schedule: &[usize]) -> Option<Vec<T>> {
    if shuffled.len() != schedule.len() {
        return None;
    }
    let mut out: Vec<Option<T>> = (0..schedule.len()).map(|_| None).collect();
    for (item, &orig) in shuffled.into_iter().zip(schedule) {
        if orig >= out.len() || out[orig].is_some() {
            return None;
        }
        out[orig] = Some(item);
    }
    out.into_iter().collect()
}

fn results_in_piece_order(
    results: Vec<(usize, u32)>,
    schedule: &[usize],
) -> Result<Vec<(usize, u32)>, CopyPortError> {
    if results
        .iter()
        .any(|(sent, _)| *sent == TRANSFER_FAILED_SENTINEL)
    {
        return Err(CopyPortError::TransferFailed(
            "gRPC/RDMA transfer failed (check Rust logs)".into(),
        ));
    }
    restore_piece_order(results, schedule).ok_or_else(|| {
        CopyPortError::TransferFailed("shuffled download result count/order mismatch".into())
    })
}

pub(crate) fn combine_transfer_checksums(
    results: &[(usize, u32)],
    expected: &[usize],
) -> Result<u32, CopyPortError> {
    let mut checksum = 1u32;
    for (&(sent, sum), &want) in results.iter().zip(expected) {
        if sent == TRANSFER_FAILED_SENTINEL {
            return Err(CopyPortError::TransferFailed(
                "gRPC/RDMA transfer failed".into(),
            ));
        }
        if sent != want {
            return Err(CopyPortError::TransferFailed(format!(
                "bad sent size: wanted {want}, got {sent}"
            )));
        }
        adler32_combine(&mut checksum, sum, sent);
    }
    Ok(checksum)
}

struct ShardedLayout {
    piece_bytes: usize,
    ownership: ShardOwnership,
}

impl ShardedLayout {
    fn from_listing(name: &str, entries: &[Vec<(String, usize)>]) -> Result<Self, CopyPortError> {
        let (piece_bytes, ownership) = classify_shard_ownership(name, entries)?;
        Ok(Self {
            piece_bytes,
            ownership,
        })
    }

    fn total_pieces(&self) -> usize {
        match &self.ownership {
            ShardOwnership::Sharded { owner, .. } => owner.len(),
            ShardOwnership::Replicated { total_pieces } => *total_pieces,
        }
    }

    fn listed_bytes(&self) -> usize {
        self.piece_bytes * self.total_pieces()
    }

    fn check_buffer_size(&self, name: &str, buf_len: usize) -> Result<(), CopyPortError> {
        let listed = self.listed_bytes();
        if listed != buf_len {
            match &self.ownership {
                ShardOwnership::Sharded {
                    owner,
                    pieces_per_rank,
                } => {
                    let n_ranks = if *pieces_per_rank == 0 {
                        0
                    } else {
                        owner.len() / pieces_per_rank
                    };
                    log::error!(
                        "copy_port: {name} size mismatch listed={listed} buf={buf_len} \
                         shard_owners={n_ranks} pieces_per_rank={pieces_per_rank}"
                    );
                }
                ShardOwnership::Replicated { total_pieces } => {
                    log::error!(
                        "copy_port: {name} size mismatch listed={listed} buf={buf_len} \
                         replicated_pieces={total_pieces}"
                    );
                }
            }
            return Err(CopyPortError::SizeMismatch {
                key: name.to_string(),
                listed,
                requested: buf_len,
            });
        }
        Ok(())
    }
}

async fn spawn_striped_trainer_downloads(
    layout: &ShardedLayout,
    prefix: &str,
    name: &str,
    channels: &[Channel],
    endpoints: &[Endpoint],
    buf: &mut [u8],
    connections: usize,
    progress: &Option<Arc<dyn DownloadProgress>>,
) -> Result<(Vec<TransferFuture>, Vec<usize>, Vec<usize>), CopyPortError> {
    let ShardOwnership::Sharded { owner, .. } = &layout.ownership else {
        return Err(CopyPortError::Other(
            "trainer striping requires shard owners".into(),
        ));
    };
    let piece = layout.piece_bytes;
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for j in 0..layout.total_pieces() {
        let key = format!("{name}/c/{j}/0");
        let Some(&source) = owner.get(&key) else {
            return Err(CopyPortError::NotFound { key });
        };
        if let Some((last, _, end)) = runs.last_mut()
            && *last == source
        {
            *end += piece;
        } else {
            runs.push((source, j * piece, (j + 1) * piece));
        }
    }
    let groups = join_all(channels.iter().enumerate().map(|(source, primary)| {
        let max_bytes = runs
            .iter()
            .filter(|r| r.0 == source)
            .map(|r| r.2 - r.1)
            .max()
            .unwrap_or(0);
        let owned: Vec<_> = owner
            .iter()
            .filter(|(_, i)| **i == source)
            .map(|(key, _)| key)
            .collect();
        async move {
            let mut group = vec![primary.clone()];
            let Some(endpoint) = endpoints.get(source) else {
                return group;
            };
            let extras = join_all((1..connections.min(max_bytes)).map(|_| {
                let owned = &owned;
                async move {
                    let candidate = match endpoint.connect().await {
                        Ok(channel) => channel,
                        Err(e) => {
                            log::warn!("copy_port: extra trainer connection {source} failed ({e})");
                            return None;
                        }
                    };
                    match list_entries(slice::from_ref(&candidate), prefix).await {
                        Ok(lists)
                            if lists.first().is_some_and(|list| {
                                owned.iter().all(|key| {
                                    list.iter().any(|(k, size)| k == *key && *size == piece)
                                })
                            }) =>
                        {
                            Some(candidate)
                        }
                        other => {
                            log::warn!(
                                "copy_port: dropping extra trainer connection {source}: \
                                 missing owned files for {prefix}{name} or listing failed ({:?})",
                                other.err()
                            );
                            None
                        }
                    }
                }
            }))
            .await;
            group.extend(extras.into_iter().flatten());
            group
        }
    }))
    .await;

    let mut futures: Vec<TransferFuture> = Vec::new();
    let mut expected = Vec::new();
    for (source, start, end) in runs {
        let bytes = end - start;
        let count = groups[source].len().min(bytes);
        let boundary = |i: usize| start + i * (bytes / count) + i.min(bytes % count);
        for (connection, channel) in groups[source].iter().take(count).enumerate() {
            let (lo, hi) = (boundary(connection), boundary(connection + 1));
            let mut names = Vec::new();
            let mut offsets = Vec::new();
            let mut sizes = Vec::new();
            for j in lo / piece..hi.div_ceil(piece) {
                let offset = lo.saturating_sub(j * piece);
                names.push(format!("{prefix}{name}/c/{j}/0").into_bytes());
                offsets.push(offset);
                sizes.push((hi - j * piece).min(piece) - offset);
            }
            expected.push(hi - lo);
            let b: &'static mut [u8] = unsafe { mem::transmute(&mut buf[lo..hi]) };
            futures.push(Box::pin(send_entries_reporting(
                channel.clone(),
                names,
                offsets,
                sizes,
                b,
                format!("source={source} connection={connection}"),
                ProgressAt::at(progress, lo),
                #[cfg(target_os = "linux")]
                (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
            )));
        }
    }
    let schedule = shuffle_sharded_schedule(futures.len(), name);
    let mut futures: Vec<_> = futures.into_iter().map(Some).collect();
    let shuffled = schedule
        .iter()
        .map(|&i| futures[i].take().unwrap())
        .collect();
    Ok((shuffled, expected, schedule))
}

async fn spawn_sharded_downloads(
    layout: &ShardedLayout,
    prefix: &str,
    name: &str,
    channels: &[Channel],
    buf: &mut [u8],
    progress: &Option<Arc<dyn DownloadProgress>>,
) -> Result<(Vec<TransferFuture>, Vec<usize>, Vec<usize>), CopyPortError> {
    let name_prefix = format!("{name}/c/");
    let piece = layout.piece_bytes;

    let ranges: Vec<(usize, usize, usize)> = match &layout.ownership {
        ShardOwnership::Sharded {
            owner,
            pieces_per_rank,
        } => {
            let step = *pieces_per_rank;
            if step == 0 {
                return Err(CopyPortError::Other("pieces_per_rank is 0".into()));
            }
            let n_ranks = owner.len() / step;
            if n_ranks == 0 {
                return Err(CopyPortError::NotFound {
                    key: name.to_string(),
                });
            }
            let mut ranges = Vec::with_capacity(n_ranks);
            for i in 0..n_ranks {
                let shard_idx = i * step;
                let key = format!("{name_prefix}{shard_idx}/0");
                let Some(&idx) = owner.get(&key) else {
                    return Err(CopyPortError::NotFound { key });
                };
                ranges.push((idx, shard_idx, shard_idx + step));
            }
            ranges
        }
        ShardOwnership::Replicated { total_pieces } => {
            replicated_send_ranges(*total_pieces, channels.len(), peer_send_max_pieces(piece))
        }
    };

    let schedule = match &layout.ownership {
        ShardOwnership::Sharded { .. } => shuffle_sharded_schedule(ranges.len(), name),
        ShardOwnership::Replicated { .. } => (0..ranges.len()).collect(),
    };
    if matches!(layout.ownership, ShardOwnership::Sharded { .. }) && ranges.len() > 1 {
        log::info!(
            "copy_port: shard schedule shuffled name={name} n={} first={:?}",
            schedule.len(),
            &schedule[..schedule.len().min(8)]
        );
    }

    #[cfg(target_os = "linux")]
    let (contexts, devicez, mrx) = {
        use crate::emb_table::register_tensor_for_download;
        use crate::ibverbs_util::make_ib_contexts;

        let rdma = matches!(layout.ownership, ShardOwnership::Sharded { .. });
        let contexts = Arc::new(if rdma {
            make_ib_contexts().map_err(|e| CopyPortError::Other(format!("RDMA context: {e}")))?
        } else {
            Vec::new()
        });
        let devicez = Arc::new(
            (0..ranges.len())
                .map(|i| {
                    if contexts.is_empty() {
                        Vec::new()
                    } else {
                        vec![i * contexts.len() / ranges.len()]
                    }
                })
                .collect::<Vec<_>>(),
        );
        let mrx = if contexts.is_empty() {
            (0..ranges.len()).map(|_| Arc::new(Vec::new())).collect()
        } else {
            let step = ranges.first().map(|(_, a, b)| b - a).unwrap_or(0);
            let b: &'static mut [u8] = unsafe { mem::transmute(&mut buf[..]) };
            register_tensor_for_download(
                piece,
                ranges.len(),
                step,
                contexts.clone(),
                devicez.clone(),
                b,
            )
            .await
            .map_err(|s| CopyPortError::Other(s.message().to_string()))?
        };
        (contexts, devicez, mrx)
    };

    let expected: Vec<usize> = ranges.iter().map(|&(_, a, b)| (b - a) * piece).collect();
    let mut futures = Vec::with_capacity(ranges.len());
    for &i in &schedule {
        let (idx, a, b) = ranges[i];
        let n = b - a;
        let slice = &mut buf[a * piece..b * piece];
        let slice: &'static mut [u8] =
            unsafe { mem::transmute(slice::from_raw_parts_mut(slice.as_mut_ptr(), slice.len())) };
        let fut: TransferFuture = Box::pin(send_entries_reporting(
            channels[idx].clone(),
            (a..b)
                .map(|j| format!("{prefix}{name_prefix}{j}/0").into_bytes())
                .collect(),
            vec![0; n],
            vec![piece; n],
            slice,
            format!("rank={i}"),
            ProgressAt::at(progress, a * piece),
            #[cfg(target_os = "linux")]
            (devicez[i].clone(), contexts.clone(), mrx[i].clone()),
        ));
        futures.push(fut);
    }
    Ok((futures, expected, schedule))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_from_trailing_slash_path() {
        let p =
            checkpoint_prefix_from_path("/ckpt/run/elapsed_samples_000000000012345678/abc123def/")
                .unwrap();
        assert_eq!(p, "elapsed_samples_000000000012345678/abc123def/");
    }

    #[test]
    fn prefix_from_no_trailing_slash() {
        let p =
            checkpoint_prefix_from_path("/ckpt/run/elapsed_samples_000000000012345678/abc123def")
                .unwrap();
        assert_eq!(p, "elapsed_samples_000000000012345678/abc123def/");
    }

    #[test]
    fn prefix_rejects_short_path() {
        assert!(checkpoint_prefix_from_path("only_one").is_err());
    }

    #[test]
    fn set_or_check_first_then_match() {
        let mut slot = 0usize;
        set_or_check(&mut slot, 64, "piece").unwrap();
        assert_eq!(slot, 64);
        set_or_check(&mut slot, 64, "piece").unwrap();
        assert!(set_or_check(&mut slot, 32, "piece").is_err());
        assert!(set_or_check(&mut slot, 0, "piece").is_err());
    }

    fn ckpt_listing(elapsed: u64, bundled: bool) -> Vec<(String, usize)> {
        let prefix = format!("elapsed_samples_{elapsed:018}/run");
        let mut l = vec![(format!("{prefix}/dense/w"), 8)];
        if bundled {
            l.push((format!("{prefix}/{BUNDLE_MANIFEST_NAME}"), 128));
        }
        l
    }

    fn ckpt_prefix(elapsed: u64) -> String {
        format!("elapsed_samples_{elapsed:018}/run")
    }

    #[test]
    fn full_prefixes_are_newest_first_and_require_every_active_channel() {
        let mut a = ckpt_listing(10, true);
        a.extend(ckpt_listing(20, true));
        a.extend(ckpt_listing(30, true));
        let mut b = ckpt_listing(10, true);
        b.extend(ckpt_listing(20, true));
        let entries = vec![a, b, Vec::new()];
        assert_eq!(
            full_prefixes_newest_first(&entries),
            vec![ckpt_prefix(20), ckpt_prefix(10)]
        );
        assert_eq!(newest_full_prefix(&entries), ckpt_prefix(20));
    }

    #[test]
    fn bundle_discovery_picks_newest_bundled_checkpoint() {
        let mut l = ckpt_listing(10, true);
        l.extend(ckpt_listing(20, true));
        let entries = vec![l.clone(), l];
        let (prefix, elapsed) = select_bundle_checkpoint(&entries, 0).unwrap().unwrap();
        assert_eq!(prefix, ckpt_prefix(20));
        assert_eq!(elapsed, 20);
    }

    #[test]
    fn bundle_discovery_falls_back_past_unbundled_newest() {
        let mut a = ckpt_listing(10, true);
        a.extend(ckpt_listing(20, true));
        a.extend(ckpt_listing(30, true));
        let mut b = ckpt_listing(10, true);
        b.extend(ckpt_listing(20, true));
        b.extend(ckpt_listing(30, false));
        let entries = vec![a, b];
        let (prefix, elapsed) = select_bundle_checkpoint(&entries, 0).unwrap().unwrap();
        assert_eq!(prefix, ckpt_prefix(20));
        assert_eq!(elapsed, 20);
    }

    #[test]
    fn bundle_discovery_never_returns_older_than_current() {
        let mut l = ckpt_listing(10, true);
        l.extend(ckpt_listing(20, true));
        l.extend(ckpt_listing(30, false));
        let entries = vec![l.clone(), l];
        match select_bundle_checkpoint(&entries, 20) {
            Err(CopyPortError::NotFound { key }) => {
                assert_eq!(key, format!("{}/{BUNDLE_MANIFEST_NAME}", ckpt_prefix(30)));
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn bundle_discovery_none_when_nothing_newer() {
        let mut l = ckpt_listing(10, true);
        l.extend(ckpt_listing(20, true));
        let entries = vec![l.clone(), l];
        assert!(select_bundle_checkpoint(&entries, 20).unwrap().is_none());
        assert!(select_bundle_checkpoint(&entries, 25).unwrap().is_none());
    }

    #[test]
    fn bundle_discovery_rejects_zero_size_manifest() {
        let prefix = ckpt_prefix(10);
        let l = vec![
            (format!("{prefix}/dense/w"), 8),
            (format!("{prefix}/{BUNDLE_MANIFEST_NAME}"), 0),
        ];
        let entries = vec![l.clone(), l];
        assert!(matches!(
            select_bundle_checkpoint(&entries, 0),
            Err(CopyPortError::NotFound { .. })
        ));
    }

    fn listing(pieces: &[usize], size: usize) -> Vec<(String, usize)> {
        pieces
            .iter()
            .map(|j| (format!("emb/c/{j}/0"), size))
            .collect()
    }

    #[test]
    fn classify_sharded_owners() {
        let entries = vec![listing(&[0, 1], 64), listing(&[2, 3], 64)];
        let (piece, ownership) = classify_shard_ownership("emb", &entries).unwrap();
        assert_eq!(piece, 64);
        let ShardOwnership::Sharded {
            owner,
            pieces_per_rank,
        } = ownership
        else {
            panic!("expected sharded");
        };
        assert_eq!(pieces_per_rank, 2);
        assert_eq!(owner["emb/c/2/0"], 1);
        assert_eq!(owner["emb/c/0/0"], 0);
    }

    #[test]
    fn classify_replicated_peers() {
        let all = [0, 1, 2, 3, 4];
        let entries = vec![listing(&all, 64), listing(&all, 64), listing(&all, 64)];
        let (piece, ownership) = classify_shard_ownership("emb", &entries).unwrap();
        assert_eq!(piece, 64);
        assert!(matches!(
            ownership,
            ShardOwnership::Replicated { total_pieces: 5 }
        ));
    }

    #[test]
    fn classify_replicated_requires_prefiltering_empty_channels() {
        let all = [0, 1, 2, 3, 4];
        let with_stale = vec![listing(&all, 64), vec![], listing(&all, 64)];
        assert!(classify_shard_ownership("emb", &with_stale).is_err());
        let degraded = vec![listing(&all, 64), vec![]];
        let (_, ownership) = classify_shard_ownership("emb", &degraded).unwrap();
        assert!(matches!(ownership, ShardOwnership::Sharded { .. }));
        let filtered = vec![listing(&all, 64), listing(&all, 64)];
        let (_, ownership) = classify_shard_ownership("emb", &filtered).unwrap();
        assert!(matches!(
            ownership,
            ShardOwnership::Replicated { total_pieces: 5 }
        ));
    }

    #[test]
    fn classify_single_channel_stays_sharded() {
        let entries = vec![listing(&[0, 1, 2], 64)];
        let (_, ownership) = classify_shard_ownership("emb", &entries).unwrap();
        assert!(matches!(
            ownership,
            ShardOwnership::Sharded {
                pieces_per_rank: 3,
                ..
            }
        ));
    }

    #[test]
    fn classify_rejects_inconsistent_ownership() {
        let entries = vec![listing(&[0, 1], 64), listing(&[1, 2], 64)];
        assert!(classify_shard_ownership("emb", &entries).is_err());
        let entries = vec![listing(&[0], 64), listing(&[1], 32)];
        assert!(classify_shard_ownership("emb", &entries).is_err());
        assert!(classify_shard_ownership("emb", &[vec![], vec![]]).is_err());
    }

    fn replicated_block_range(total: usize, n: usize, i: usize) -> (usize, usize) {
        let step = total.div_ceil(n);
        (
            std::cmp::min(total, i * step),
            std::cmp::min(total, (i + 1) * step),
        )
    }

    #[test]
    fn replicated_block_ranges_cover_all_pieces() {
        assert_eq!(replicated_block_range(5, 2, 0), (0, 3));
        assert_eq!(replicated_block_range(5, 2, 1), (3, 5));
        assert_eq!(replicated_block_range(2, 4, 1), (1, 2));
        assert_eq!(replicated_block_range(2, 4, 2), (2, 2));
        for (total, n) in [(1, 1), (7, 3), (256, 4), (3, 8)] {
            let mut covered = 0;
            for i in 0..n {
                let (a, b) = replicated_block_range(total, n, i);
                assert_eq!(a, covered.min(total));
                covered = b.max(covered);
            }
            assert_eq!(covered, total);
        }
    }

    #[test]
    fn combine_checksums_validates_sizes() {
        assert!(combine_transfer_checksums(&[(64, 1)], &[64]).is_ok());
        assert!(combine_transfer_checksums(&[(63, 1)], &[64]).is_err());
        assert!(combine_transfer_checksums(&[(TRANSFER_FAILED_SENTINEL, 1)], &[64]).is_err());
    }

    #[test]
    fn shuffle_schedule_is_stable_permutation() {
        let a = shuffle_sharded_schedule(32, "emb_table");
        let b = shuffle_sharded_schedule(32, "emb_table");
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<_>>());
        let c = shuffle_sharded_schedule(32, "post_embeddings");
        assert_ne!(a, c);
        assert_ne!(a, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn restore_piece_order_inverts_schedule() {
        let schedule = vec![3, 0, 2, 1];
        let shuffled = vec!['d', 'a', 'c', 'b'];
        assert_eq!(
            restore_piece_order(shuffled, &schedule).unwrap(),
            vec!['a', 'b', 'c', 'd']
        );
        assert!(restore_piece_order(vec![1, 2], &[0, 1, 2]).is_none());
        assert!(restore_piece_order(vec![1, 2, 3], &[0, 0, 1]).is_none());
    }

    #[test]
    fn combine_after_restore_matches_piece_order() {
        let piece_order = vec![(10, 11u32), (10, 22), (10, 33)];
        let expected = vec![10usize, 10, 10];
        let direct = combine_transfer_checksums(&piece_order, &expected).unwrap();
        let schedule = vec![2, 0, 1];
        let shuffled = vec![piece_order[2], piece_order[0], piece_order[1]];
        let restored = restore_piece_order(shuffled.clone(), &schedule).unwrap();
        assert_eq!(
            combine_transfer_checksums(&restored, &expected).unwrap(),
            direct
        );
        assert_ne!(
            combine_transfer_checksums(&shuffled, &expected).unwrap(),
            direct
        );
    }

    #[test]
    fn replicated_send_ranges_chunk_within_blocks() {
        assert_eq!(
            replicated_send_ranges(10, 1, 4),
            vec![(0, 0, 4), (0, 4, 8), (0, 8, 10)]
        );
        assert_eq!(
            replicated_send_ranges(5, 2, 2),
            vec![(0, 0, 2), (1, 2, 4), (0, 4, 5)]
        );
        assert_eq!(replicated_send_ranges(2, 4, 8), vec![(0, 0, 1), (1, 1, 2)]);
        assert_eq!(
            replicated_send_ranges(3, 1, 0),
            vec![(0, 0, 1), (0, 1, 2), (0, 2, 3)]
        );
    }

    #[test]
    fn replicated_send_ranges_round_robin_spans_all_channels() {
        for (total, n, cap) in [(30, 2, 5), (30, 8, 2), (256, 4, 8), (17, 3, 1)] {
            let ranges = replicated_send_ranges(total, n, cap);
            assert!(ranges.len() >= n, "every channel must get work");
            for w in ranges.windows(n) {
                let mut seen: Vec<usize> = w.iter().map(|&(c, _, _)| c).collect();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(seen.len(), n, "window must span all {n} channels");
            }
        }
        let ranges = replicated_send_ranges(30, 3, usize::MAX);
        let channels: Vec<usize> = ranges.iter().map(|&(c, _, _)| c).collect();
        assert_eq!(channels, vec![0, 1, 2]);
    }

    #[test]
    fn replicated_send_ranges_golden_uncapped_matches_block_range() {
        for (total, n) in [(1, 1), (7, 3), (256, 4), (3, 8), (230, 2)] {
            let uncapped = replicated_send_ranges(total, n, usize::MAX);
            let blocks: Vec<_> = (0..n)
                .map(|i| {
                    let (a, b) = replicated_block_range(total, n, i);
                    (i, a, b)
                })
                .filter(|(_, a, b)| a < b)
                .collect();
            assert_eq!(uncapped, blocks);
        }
    }

    #[test]
    fn replicated_send_ranges_cover_all_pieces_in_order() {
        for (total, n, cap) in [(1, 1, 1), (7, 3, 2), (256, 4, 8), (3, 8, 1), (230, 2, 7)] {
            let ranges = replicated_send_ranges(total, n, cap);
            let mut covered = 0;
            for &(_, a, b) in &ranges {
                assert_eq!(a, covered, "ranges must be ascending and gap-free");
                assert!(b > a && b - a <= cap.max(1));
                covered = b;
            }
            assert_eq!(covered, total);
        }
    }

    #[test]
    fn peer_send_max_pieces_bounds() {
        assert_eq!(peer_send_max_pieces(1_920_004_096), 8);
        assert_eq!(peer_send_max_pieces(usize::MAX), 1);
        assert_eq!(peer_send_max_pieces(0), *PEER_SEND_MAX_BYTES);
    }

    #[test]
    fn choose_prefix_respects_target() {
        let entries = vec![
            vec![("elapsed_samples_1/run/x".to_string(), 1)],
            vec![("elapsed_samples_1/run/x".to_string(), 1)],
        ];
        assert_eq!(
            choose_prefix(Some("elapsed_samples_1/run"), &entries).unwrap(),
            "elapsed_samples_1/run"
        );
        assert_eq!(
            choose_prefix(Some("elapsed_samples_1/run/"), &entries).unwrap(),
            "elapsed_samples_1/run"
        );
        let partial = vec![entries[0].clone(), vec![]];
        assert!(choose_prefix(Some("elapsed_samples_1/run"), &partial).is_err());
    }

    #[test]
    fn re_served_pieces_split_to_at_most_a_gib() {
        let layout = |piece_bytes, pieces| ShardLayout {
            piece_bytes,
            pieces,
        };
        assert_eq!(
            layout(960_002_048, 512),
            layout(7_680_016_384, 64).split(PEER_PIECE_MAX_BYTES)
        );
        assert_eq!(layout(1 << 20, 3), layout(1 << 20, 3).split(1 << 30));
        assert_eq!(layout(4099, 2), layout(4099, 2).split(1024));
        assert_eq!(layout(1024, 8), layout(4096, 2).split(1024));
    }

    #[test]
    fn listed_pieces_of_one_tensor() {
        let entries = |names: &[(&str, usize)]| -> Vec<(String, usize)> {
            names.iter().map(|&(n, s)| (n.to_string(), s)).collect()
        };
        assert_eq!(
            (8, vec![true, false, true]),
            listed_pieces(
                "emb",
                &entries(&[
                    ("emb/c/2/0", 8),
                    ("emb/c/0/0", 8),
                    ("emb_other/c/1/0", 4),
                    ("params.w/c/0", 3),
                    ("checksums.0.json", 2),
                ])
            )
        );
        assert_eq!(
            (0, vec![]),
            listed_pieces("emb", &entries(&[("emb/c/0/0", 8), ("emb/c/1/0", 4)])),
            "pieces disagree on size"
        );
        assert_eq!((0, vec![]), listed_pieces("emb", &entries(&[])));
    }

    #[tokio::test]
    async fn a_failed_piece_goes_to_a_replica_that_is_not_failing() {
        let now = tokio::time::Instant::now();
        let source = |listed: &[bool], failures: u32, inflight: usize| {
            let channel = Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
            ProgressiveSource {
                url: String::new(),
                list: channel.clone(),
                sends: vec![channel; 4],
                next_send: 0,
                piece_bytes: 10,
                listed: listed.to_vec(),
                skipped: false,
                listing: false,
                next_list: now,
                listed_at: Some(now),
                inflight,
                failures,
                retry_at: now,
            }
        };
        let sources = [source(&[true, true], 1, 0), source(&[true, false], 0, 3)];
        assert_eq!(
            Some((1, 0)),
            next_progressive_unit(&sources, &[false, false], 10, now)
        );
        assert_eq!(
            Some((0, 1)),
            next_progressive_unit(&sources, &[true, false], 10, now),
            "only the failing replica has it"
        );
        assert_eq!(
            None,
            next_progressive_unit(&sources, &[false, false], 10, now + PROGRESSIVE_LIST_FRESH),
            "neither answered a listing lately"
        );
    }

    fn tracking_downloads(
        n: usize,
        bytes: usize,
        hold: Duration,
        in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        peak: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        starts: std::sync::Arc<std::sync::Mutex<Vec<Option<Instant>>>>,
    ) -> Vec<TransferFuture> {
        (0..n)
            .map(|i| {
                let in_flight = in_flight.clone();
                let peak = peak.clone();
                let starts = starts.clone();
                Box::pin(async move {
                    starts.lock().unwrap()[i] = Some(Instant::now());
                    let cur = in_flight.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    peak.fetch_max(cur, std::sync::atomic::Ordering::SeqCst);
                    tokio::time::sleep(hold).await;
                    in_flight.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    (bytes, i as u32)
                }) as TransferFuture
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rate_limit_caps_in_flight() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = std::sync::Arc::new(std::sync::Mutex::new(vec![None; 6]));
        let futures = tracking_downloads(
            6,
            1,
            Duration::from_millis(80),
            in_flight,
            peak.clone(),
            starts,
        );
        let results = join_rate_limited(futures, 1 << 40, 2).await.unwrap();
        assert_eq!(
            results.iter().map(|r| r.1).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rate_limit_paces_between_batches() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = std::sync::Arc::new(std::sync::Mutex::new(vec![None; 4]));
        let bytes = 200 * 1024;
        let rate = 400 * 1024;
        let t0 = Instant::now();
        let results = run_downloads(
            tracking_downloads(
                4,
                bytes,
                Duration::from_millis(20),
                in_flight,
                peak.clone(),
                starts.clone(),
            ),
            Some(rate),
            Some(2),
        )
        .await
        .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(results.len(), 4);
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 2);

        let starts = starts.lock().unwrap();
        let s: Vec<Instant> = starts.iter().map(|t| t.expect("started")).collect();
        let first_batch_start = s[0].min(s[1]);
        let second_batch_start = s[2].min(s[3]);
        let between = second_batch_start.saturating_duration_since(first_batch_start);
        assert!(
            between >= Duration::from_millis(700),
            "second batch started {between:?} after the first"
        );

        let total_bytes = (4 * bytes) as f64;
        let min_elapsed = Duration::from_secs_f64(total_bytes / rate as f64 * 0.85);
        assert!(
            elapsed >= min_elapsed,
            "elapsed {elapsed:?} is below the {min_elapsed:?} floor for {total_bytes} bytes at {rate} B/s"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rate_limit_skips_remaining_on_sentinel() {
        let started = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let futures: Vec<TransferFuture> = (0..4)
            .map(|i| {
                let started = started.clone();
                Box::pin(async move {
                    started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i == 1 {
                        (TRANSFER_FAILED_SENTINEL, 0)
                    } else {
                        (1_000_000, i as u32)
                    }
                }) as TransferFuture
            })
            .collect();
        let t0 = Instant::now();
        let results = join_rate_limited(futures, 1, 2).await.unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "must not pace a failed batch"
        );
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.0 == TRANSFER_FAILED_SENTINEL));
        assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unlimited_path_spawns_all() {
        let in_flight = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let starts = std::sync::Arc::new(std::sync::Mutex::new(vec![None; 4]));
        let results = run_downloads(
            tracking_downloads(
                4,
                1,
                Duration::from_millis(80),
                in_flight,
                peak.clone(),
                starts,
            ),
            Some(0),
            Some(1),
        )
        .await
        .unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(
            peak.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "rate_limit=0 must ignore max_concurrent and spawn every future"
        );
    }

    mod scheduler {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::{mpsc, oneshot};

        #[tokio::test(start_paused = true)]
        async fn refill_preserves_order_concurrency_and_global_pacing() {
            for rate in [Some(100), Some(0), None] {
                let start = tokio::time::Instant::now();
                let events = Arc::new(Mutex::new(Vec::new()));
                let futures = (0..4)
                    .map(|index| {
                        let events = events.clone();
                        Box::pin(async move {
                            events.lock().unwrap().push((index, true, start.elapsed()));
                            if index == 0 {
                                tokio::time::sleep(Duration::from_millis(1500)).await;
                            }
                            events.lock().unwrap().push((index, false, start.elapsed()));
                            (100usize, index as u32)
                        }) as TransferFuture
                    })
                    .collect();
                let results = run_downloads(futures, rate, Some(2)).await.unwrap();
                assert_eq!(results, vec![(100, 0), (100, 1), (100, 2), (100, 3)]);
                let (mut active, mut peak) = (0, 0);
                let mut starts = [Duration::ZERO; 4];
                for &(index, started, time) in events.lock().unwrap().iter() {
                    if started {
                        starts[index] = time;
                        active += 1;
                        peak = peak.max(active);
                    } else {
                        active -= 1;
                    }
                }
                if rate == Some(100) {
                    assert_eq!(starts.map(|time| time.as_millis()), [0, 0, 1000, 3000]);
                    assert_eq!(start.elapsed(), Duration::from_secs(4));
                    assert_eq!(peak, 2);
                } else {
                    assert_eq!(starts, [Duration::ZERO; 4]);
                }
                assert_eq!(active, 0);
            }
        }

        #[tokio::test(start_paused = true)]
        async fn failure_stops_admissions_and_drains_active_work() {
            for panic in [false, true] {
                let (started, mut starts) = mpsc::unbounded_channel();
                let (finished, mut finishes) = mpsc::unbounded_channel();
                let mut finish = Vec::new();
                let futures = (0..4)
                    .map(|index| {
                        let (tx, rx) = oneshot::channel();
                        finish.push(Some(tx));
                        let (started, finished) = (started.clone(), finished.clone());
                        Box::pin(async move {
                            scopeguard::defer! { finished.send(index).unwrap(); }
                            started.send(index).unwrap();
                            (rx.await.expect("test transfer panicked"), index as u32)
                        }) as TransferFuture
                    })
                    .collect();
                let start = tokio::time::Instant::now();
                let mut download = Box::pin(run_downloads(futures, Some(1), Some(3)));
                assert!(futures::poll!(&mut download).is_pending());
                for _ in 0..3 {
                    starts.recv().await.unwrap();
                }
                finish[0].take().unwrap().send(100).unwrap();
                assert_eq!(finishes.recv().await, Some(0));
                assert!(futures::poll!(&mut download).is_pending());
                if panic {
                    drop(finish[1].take());
                } else {
                    finish[1]
                        .take()
                        .unwrap()
                        .send(TRANSFER_FAILED_SENTINEL)
                        .unwrap();
                }
                assert_eq!(finishes.recv().await, Some(1));
                assert!(futures::poll!(&mut download).is_pending());
                assert!(starts.try_recv().is_err());
                finish[2].take().unwrap().send(100).unwrap();
                let result = download.await;
                if panic {
                    assert!(matches!(result, Err(CopyPortError::Other(_))));
                } else {
                    assert_eq!(
                        result.unwrap(),
                        vec![(100, 0), (TRANSFER_FAILED_SENTINEL, 1), (100, 2)]
                    );
                }
                assert_eq!(finishes.recv().await, Some(2));
                assert!(starts.try_recv().is_err());
                assert_eq!(start.elapsed(), Duration::ZERO);
            }
        }

        #[test]
        fn cancellation_and_shutdown_join_writers_despite_panicking_destructors() {
            struct PanicOnDrop(Arc<AtomicUsize>);
            impl Drop for PanicOnDrop {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                    panic!("test transfer destructor panicked");
                }
            }
            let (done, completed) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || {
                for multi_thread in [false, true] {
                    for mode in ["drop", "local", "shutdown"] {
                        let mut builder = if multi_thread {
                            tokio::runtime::Builder::new_multi_thread()
                        } else {
                            tokio::runtime::Builder::new_current_thread()
                        };
                        let runtime = builder.worker_threads(1).enable_all().build().unwrap();
                        let dropped = Arc::new(AtomicUsize::new(0));
                        let copied = Arc::new(AtomicUsize::new(0));
                        let (started, mut ready) = mpsc::unbounded_channel();
                        let futures = (0..3)
                            .map(|index| {
                                let started = started.clone();
                                if index == 1 {
                                    let copied = copied.clone();
                                    Box::pin(async move {
                                        let (stop, stopped) = std::sync::mpsc::channel::<()>();
                                        let worker = tokio::task::spawn_blocking(move || {
                                            started.send(()).unwrap();
                                            let _ = stopped.recv();
                                            copied.fetch_add(1, Ordering::SeqCst);
                                        });
                                        let mut worker = (
                                            stop,
                                            Box::pin(
                                                crate::emb_table::JoinOnDrop::new(worker).join(),
                                            ),
                                        );
                                        worker.1.as_mut().await.unwrap();
                                        (0usize, 0u32)
                                    }) as TransferFuture
                                } else {
                                    let guard = PanicOnDrop(dropped.clone());
                                    Box::pin(async move {
                                        let _guard = guard;
                                        started.send(()).unwrap();
                                        std::future::pending().await
                                    }) as TransferFuture
                                }
                            })
                            .collect();
                        let download = run_downloads(futures, Some(1), Some(3));
                        if mode == "shutdown" {
                            runtime.spawn(download);
                            runtime.block_on(async {
                                for _ in 0..3 {
                                    ready.recv().await.unwrap();
                                }
                            });
                            drop(runtime);
                        } else {
                            let cancel = async {
                                let mut download = Box::pin(download);
                                assert!(futures::poll!(&mut download).is_pending());
                                for _ in 0..3 {
                                    ready.recv().await.unwrap();
                                }
                                drop(download);
                            };
                            if mode == "local" {
                                runtime.block_on(tokio::task::LocalSet::new().run_until(cancel));
                            } else {
                                runtime.block_on(cancel);
                            }
                        }
                        assert_eq!(dropped.load(Ordering::SeqCst), 2);
                        assert_eq!(copied.load(Ordering::SeqCst), 1);
                    }
                }
                done.send(()).unwrap();
            });
            completed
                .recv_timeout(Duration::from_secs(10))
                .expect("cleanup hung");
            thread.join().unwrap();
        }
    }
}

#[cfg(all(test, any(not(target_os = "linux"), feature = "rdma-tests")))]
mod trainer_e2e_tests {
    use super::*;
    use crate::grpc_util::{add_ok_trailer, freeze, get_bytes_mut, make_response};
    use crate::proto_parser::{
        self, BodyKind, BytesMutProtoExt, decode_names, encode_names, parse, proto,
    };
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tonic::codegen::{Service, http};
    use tonic::{Status, body};

    const PREFIX: &str = "elapsed_samples_000000000000000042/trainer/";
    type Files = BTreeMap<String, Vec<u8>>;

    #[derive(Default)]
    struct Traffic {
        active: AtomicUsize,
        peak: AtomicUsize,
        barrier: Option<tokio::sync::Barrier>,
        fault: AtomicUsize,
    }

    #[derive(Clone)]
    struct Trainer {
        files: Arc<Files>,
        traffic: Arc<Traffic>,
        sockets: Arc<Mutex<HashSet<SocketAddr>>>,
        primary: Arc<Mutex<Option<SocketAddr>>>,
    }

    impl tonic::server::NamedService for Trainer {
        const NAME: &'static str = "copy.Copy";
    }

    impl Service<http::Request<body::Body>> for Trainer {
        type Response = http::Response<body::Body>;
        type Error = Infallible;
        type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            Ok(()).into()
        }

        fn call(&mut self, request: http::Request<body::Body>) -> Self::Future {
            let this = self.clone();
            Box::pin(async move { Ok(make_response(this.handle(request).await)) })
        }
    }

    impl Trainer {
        async fn handle(self, request: http::Request<body::Body>) -> Result<body::Body, Status> {
            use tonic::transport::server::{TcpConnectInfo, TlsConnectInfo};
            let remote = request
                .extensions()
                .get::<TcpConnectInfo>()
                .and_then(TcpConnectInfo::remote_addr)
                .or_else(|| {
                    request
                        .extensions()
                        .get::<TlsConnectInfo<TcpConnectInfo>>()
                        .and_then(|i| i.get_ref().remote_addr())
                })
                .unwrap();
            let fault = self.traffic.fault.load(Ordering::SeqCst);
            let mut out = get_bytes_mut();
            match request.uri().path() {
                "/copy.Copy/List" => {
                    parse(crate::proto![], request.into_body(), BodyKind::Request).await?;
                    let primary = *self.primary.lock().unwrap().get_or_insert(remote);
                    if fault == 1 && remote != primary {
                        return Err(Status::unavailable("extra connection cannot list"));
                    }
                    let (names, prefixes, suffixes) =
                        encode_names(self.files.keys().map(|k| k.as_bytes().to_vec()).collect());
                    out.put_string(1, &names);
                    out.put_repeated_ints(
                        [2, 3, 4],
                        [
                            &prefixes,
                            &suffixes,
                            &self.files.values().map(Vec::len).collect::<Vec<_>>(),
                        ],
                    );
                }
                "/copy.Copy/Send" => {
                    let (mut names, mut prefixes, mut suffixes, mut offsets, mut sizes) = (
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::<usize>::new(),
                        Vec::<usize>::new(),
                    );
                    parse(
                        crate::proto![
                            (1, proto_parser::bytes(&mut names)),
                            (2, proto_parser::repeated_ints(&mut prefixes)),
                            (3, proto_parser::repeated_ints(&mut suffixes)),
                            (4, proto_parser::repeated_ints(&mut offsets)),
                            (5, proto_parser::repeated_ints(&mut sizes)),
                        ],
                        request.into_body(),
                        BodyKind::Request,
                    )
                    .await?;
                    let names = decode_names(names, prefixes, suffixes)?;
                    let first_stripe = offsets.first() == Some(&0);
                    let first_name = std::str::from_utf8(&names[0]).unwrap();
                    let sharded = first_name.contains("/emb_table/")
                        || first_name.contains("/post_embeddings.embeddings/");
                    let mut data = Vec::new();
                    for ((name, offset), size) in names.into_iter().zip(offsets).zip(sizes) {
                        let name = String::from_utf8(name).unwrap();
                        let file = self
                            .files
                            .get(&name)
                            .ok_or_else(|| Status::not_found(&name))?;
                        data.extend_from_slice(
                            file.get(offset..offset + size)
                                .ok_or_else(|| Status::out_of_range(&name))?,
                        );
                    }
                    self.sockets.lock().unwrap().insert(remote);
                    let active = self.traffic.active.fetch_add(1, Ordering::SeqCst) + 1;
                    self.traffic.peak.fetch_max(active, Ordering::SeqCst);
                    if sharded && let Some(barrier) = &self.traffic.barrier {
                        tokio::time::timeout(Duration::from_secs(5), barrier.wait())
                            .await
                            .unwrap();
                    }
                    self.traffic.active.fetch_sub(1, Ordering::SeqCst);
                    if fault == 2 && first_stripe {
                        data.pop();
                    }
                    out.put_string(1, &data);
                }
                _ => return Err(Status::unimplemented("unexpected copy_port method")),
            }
            Ok(add_ok_trailer(freeze(out)))
        }
    }

    async fn serve(
        files: Files,
        traffic: Arc<Traffic>,
        tls: Option<tonic::transport::ServerTlsConfig>,
    ) -> (String, Arc<Mutex<HashSet<SocketAddr>>>, impl Drop) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let scheme = if tls.is_some() { "https" } else { "http" };
        let url = format!("{scheme}://{}", listener.local_addr().unwrap());
        let sockets = Arc::new(Mutex::new(HashSet::new()));
        let service = Trainer {
            files: Arc::new(files),
            traffic,
            sockets: sockets.clone(),
            primary: Arc::default(),
        };
        let mut builder = tonic::transport::Server::builder();
        if let Some(tls) = tls {
            builder = builder.tls_config(tls).unwrap();
        }
        let task = tokio::spawn(async move {
            builder
                .add_service(service)
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        (url, sockets, scopeguard::guard(task, |task| task.abort()))
    }

    fn tensor(name: &str, pieces: std::ops::Range<usize>, piece: usize) -> Files {
        pieces
            .map(|j| {
                (
                    format!("{PREFIX}{name}/c/{j}/0"),
                    (j * piece..(j + 1) * piece).map(|i| i as u8).collect(),
                )
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial_test::serial(copy_tls_env)]
    async fn trainer_hotswap_preserves_sources_checksums_and_shared_budget() {
        use crate::tls::{ENV_TLS_CA, ENV_TLS_SERVER_NAME};
        use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};
        xai_init_utils::init().rustls();
        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(Vec::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca = ca_params.self_signed(&ca_key).unwrap();
        let key = KeyPair::generate().unwrap();
        let cert = CertificateParams::new(vec!["copy.test".into()])
            .unwrap()
            .signed_by(&key, &ca, &ca_key)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let ca_path = dir.path().join("ca.pem");
        std::fs::write(&ca_path, ca.pem()).unwrap();
        let keys = [
            ENV_TLS_CA,
            ENV_TLS_SERVER_NAME,
            "COPY_PORT_TRAINER_CONNS_PER_SOURCE",
            "COPY_PORT_PEER_CONNS_PER_SOURCE",
        ];
        let saved = keys.map(|key| (key, std::env::var_os(key)));
        let _restore = scopeguard::guard(saved, |saved| {
            for (key, value) in saved {
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(key, v),
                        None => std::env::remove_var(key),
                    }
                }
            }
        });
        unsafe {
            std::env::set_var(ENV_TLS_CA, &ca_path);
            std::env::set_var(ENV_TLS_SERVER_NAME, "copy.test");
            std::env::set_var("COPY_PORT_PEER_CONNS_PER_SOURCE", "16");
        }
        for (connections, rate) in [(1, None), (2, None), (4, Some(64))] {
            unsafe {
                if connections == 1 {
                    std::env::remove_var(keys[2]);
                } else {
                    std::env::set_var(keys[2], connections.to_string());
                }
            }
            let traffic = Arc::new(Traffic {
                barrier: Some(tokio::sync::Barrier::new(2)),
                ..Traffic::default()
            });
            let files = |pieces: std::ops::Range<usize>| {
                [
                    tensor("emb_table", pieces.clone(), 7),
                    tensor("post_embeddings.embeddings", pieces, 5),
                    tensor("dense", 0..1, 3),
                    [(
                        format!("{PREFIX}checksums.0.json"),
                        b"{\"created_timestamp\":42}".to_vec(),
                    )]
                    .into_iter()
                    .collect(),
                ]
                .into_iter()
                .flatten()
                .collect()
            };
            let tls = tonic::transport::ServerTlsConfig::new().identity(
                tonic::transport::Identity::from_pem(cert.pem(), key.serialize_pem()),
            );
            let (a, a_sockets, _a) = serve(files(2..4), traffic.clone(), Some(tls)).await;
            let (driver, _, _driver) = serve(Files::new(), traffic.clone(), None).await;
            let (b, b_sockets, _b) = serve(files(0..2), traffic.clone(), None).await;
            let (mut dense, mut emb, mut pe) = (vec![255; 3], vec![255; 28], vec![255; 20]);
            let start = Instant::now();
            let (meta, emb_sum, pe_sum) = download_dense_and_embeddings(
                0,
                &format!("{a},{driver},{b}"),
                &mut [TensorBuf {
                    key: "dense".into(),
                    buf: &mut dense,
                }],
                &mut emb,
                Some(&mut pe),
                rate,
                Some(connections.max(2)),
            )
            .await
            .unwrap();
            assert_eq!(meta.prefix, PREFIX.trim_end_matches('/'));
            assert_eq!(meta.created_timestamp, 42.0);
            assert_eq!(dense, vec![0, 1, 2]);
            assert_eq!(emb, (0..28).collect::<Vec<u8>>());
            assert_eq!(pe, (0..20).collect::<Vec<u8>>());
            assert_eq!(emb_sum, simd_adler32::adler32(&emb.as_slice()));
            assert_eq!(pe_sum, Some(simd_adler32::adler32(&pe.as_slice())));
            for sockets in [a_sockets, b_sockets] {
                assert_eq!(sockets.lock().unwrap().len(), 2 * connections - 1);
            }
            assert!(traffic.peak.load(Ordering::SeqCst) <= connections.max(2));
            assert_eq!(traffic.active.load(Ordering::SeqCst), 0);
            if rate.is_some() {
                assert!(start.elapsed() >= Duration::from_millis(750));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial_test::serial(copy_tls_env)]
    async fn trainer_extra_listing_fallback_and_short_transfer_retry() {
        let traffic = Arc::new(Traffic::default());
        let (url, sockets, _server) =
            serve(tensor("emb_table", 0..2, 7), traffic.clone(), None).await;
        let mut buf = vec![255; 14];
        for fault in [1, 2, 0] {
            traffic.fault.store(fault, Ordering::SeqCst);
            sockets.lock().unwrap().clear();
            buf.fill(255);
            let result = download_embedding_table_with_conns(
                PREFIX,
                &url,
                "emb_table",
                &mut buf,
                None,
                Some(2),
                16,
                2,
                false,
                None,
            )
            .await;
            assert_eq!(traffic.active.load(Ordering::SeqCst), 0);
            if fault == 2 {
                assert!(matches!(result, Err(CopyPortError::TransferFailed(_))));
            } else {
                assert_eq!(result.unwrap().0, simd_adler32::adler32(&buf.as_slice()));
                assert_eq!(buf, (0..14).collect::<Vec<u8>>());
            }
            assert_eq!(
                sockets.lock().unwrap().len(),
                if fault == 1 { 1 } else { 2 }
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[serial_test::serial(copy_tls_env)]
    async fn single_peer_downloads_as_a_replica() {
        for (replicas, sends_from) in [(false, 1), (true, 4)] {
            let traffic = Arc::new(Traffic::default());
            let (url, sockets, _server) =
                serve(tensor("emb_table", 0..4, 7), traffic.clone(), None).await;
            let mut buf = vec![255; 28];
            let (checksum, layout) = download_embedding_table_with_conns(
                PREFIX,
                &url,
                "emb_table",
                &mut buf,
                None,
                Some(4),
                4,
                1,
                replicas,
                None,
            )
            .await
            .unwrap();
            assert_eq!((0..28).collect::<Vec<u8>>(), buf);
            assert_eq!(simd_adler32::adler32(&buf.as_slice()), checksum);
            assert_eq!(
                ShardLayout {
                    piece_bytes: 7,
                    pieces: 4
                },
                layout
            );
            assert_eq!(sends_from, sockets.lock().unwrap().len());
        }
    }
}

#[cfg(all(test, target_os = "linux", feature = "rdma-tests"))]
mod p2p_e2e_tests {
    use super::*;
    use crate::copy::{CopyService, ManifestSpec, Sender};
    use crate::copy_manifest::PeerServeState;
    use bytes::Bytes;
    use simd_adler32::adler32;

    const PREFIX: &str = "elapsed_samples_000000000000000042/testrun/";
    const PIECE: usize = 1000;
    const PIECES: usize = 5;

    async fn spawn_peer(backing: &std::path::Path) -> String {
        let sender = Arc::new(Sender::new(12).with_peer_serve(4, None));
        sender
            .publish_manifest(
                (0..PIECES)
                    .map(|j| ManifestSpec {
                        name: format!("{PREFIX}emb/c/{j}/0"),
                        file: backing.to_str().unwrap().to_string(),
                        offset: j * PIECE,
                        size: PIECE,
                    })
                    .collect(),
                vec![],
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        format!("127.0.0.1:{}", addr.port())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replicated_download_from_two_peers() {
        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 249) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let peer_a = spawn_peer(&path).await;
        let peer_b = spawn_peer(&path).await;
        let urls = format!("{peer_a},{peer_b}");

        let mut buf = vec![0u8; PIECE * PIECES];
        let checksum = download_embedding_table(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(buf, data);
        assert_eq!(checksum, adler32(&&data[..]));
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replicated_download_multi_connection() {
        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 251) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_mc_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let peer_a = spawn_peer(&path).await;
        let peer_b = spawn_peer(&path).await;
        let urls = format!("{peer_a},{peer_b}");

        let mut buf = vec![0u8; PIECE * PIECES];
        let (checksum, layout) = download_embedding_table_with_conns(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
            2,
            1,
            true,
            None,
        )
        .await
        .unwrap();

        assert_eq!(buf, data);
        assert_eq!(checksum, adler32(&&data[..]));
        assert_eq!(
            ShardLayout {
                piece_bytes: PIECE,
                pieces: PIECES
            },
            layout
        );
        let _ = std::fs::remove_file(&path);
    }

    async fn spawn_stale_peer(backing: &std::path::Path) -> String {
        let sender = Arc::new(Sender::new(12).with_peer_serve(4, None));
        sender
            .publish_manifest(
                vec![ManifestSpec {
                    name: "elapsed_samples_000000000000000041/oldrun/emb/c/0/0".to_string(),
                    file: backing.to_str().unwrap().to_string(),
                    offset: 0,
                    size: PIECE,
                }],
                vec![],
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        format!("127.0.0.1:{}", addr.port())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replicated_multi_connection_excludes_stale_source() {
        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 253) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_stale_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let peer_a = spawn_peer(&path).await;
        let peer_b = spawn_peer(&path).await;
        let stale = spawn_stale_peer(&path).await;
        let urls = format!("{peer_a},{stale},{peer_b}");

        let mut buf = vec![0u8; PIECE * PIECES];
        let (checksum, layout) = download_embedding_table_with_conns(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
            2,
            1,
            true,
            None,
        )
        .await
        .unwrap();

        assert_eq!(buf, data);
        assert_eq!(checksum, adler32(&&data[..]));
        assert_eq!(
            ShardLayout {
                piece_bytes: PIECE,
                pieces: PIECES
            },
            layout
        );
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial_test::serial(copy_tls_env)]
    async fn replicated_download_over_tls() {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};

        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(Vec::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let server_key = KeyPair::generate().unwrap();
        let server_cert = CertificateParams::new(vec!["copy.test".to_string()])
            .unwrap()
            .signed_by(&server_key, &ca_cert, &ca_key)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, pem: &str| {
            let p = dir.path().join(name);
            std::fs::write(&p, pem).unwrap();
            p.to_str().unwrap().to_string()
        };
        let (ca, cert, key) = (
            write("ca.crt", &ca_cert.pem()),
            write("tls.crt", &server_cert.pem()),
            write("tls.key", &server_key.serialize_pem()),
        );

        let data: Vec<u8> = (0..PIECE * PIECES).map(|i| (i % 249) as u8).collect();
        let path = std::env::temp_dir().join(format!("p2p_e2e_tls_{}", std::process::id()));
        std::fs::write(&path, &data).unwrap();

        let sender = Arc::new(Sender::new(12).with_peer_serve(4, None));
        sender
            .publish_manifest(
                (0..PIECES)
                    .map(|j| ManifestSpec {
                        name: format!("{PREFIX}emb/c/{j}/0"),
                        file: path.to_str().unwrap().to_string(),
                        offset: j * PIECE,
                        size: PIECE,
                    })
                    .collect(),
                vec![],
            )
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tls = crate::tls::ServerTlsOptions::from_paths(Some(cert), Some(key), None)
            .unwrap()
            .unwrap();
        tokio::spawn(
            tonic::transport::Server::builder()
                .tls_config(tls.load().unwrap())
                .unwrap()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        unsafe { std::env::set_var(crate::tls::ENV_TLS_CA, &ca) };
        unsafe { std::env::set_var(crate::tls::ENV_TLS_SERVER_NAME, "copy.test") };
        let urls = format!("https://127.0.0.1:{}", addr.port());
        let mut buf = vec![0u8; PIECE * PIECES];
        let result = download_embedding_table(
            &format!("/dev/shm/{PREFIX}"),
            &urls,
            "emb",
            &mut buf,
            None,
            None,
        )
        .await;
        unsafe { std::env::remove_var(crate::tls::ENV_TLS_CA) };
        unsafe { std::env::remove_var(crate::tls::ENV_TLS_SERVER_NAME) };

        assert_eq!(buf, data);
        assert_eq!(result.unwrap(), adler32(&&data[..]));
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial_test::serial(copy_tls_env)]
    async fn memory_manifest_round_trip() {
        let dense = Bytes::from((0..300u32).map(|i| i as u8).collect::<Vec<u8>>());
        let emb = Bytes::from(
            (0..PIECE * PIECES)
                .map(|i| (i % 247) as u8)
                .collect::<Vec<u8>>(),
        );
        let mut memory = vec![
            (format!("{PREFIX}params.w/c/0"), dense.clone()),
            (
                format!("{PREFIX}checksums.0.json"),
                Bytes::from_static(br#"{"created_timestamp":7.0}"#),
            ),
        ];
        memory.extend((0..PIECES).map(|j| {
            (
                format!("{PREFIX}emb/c/{j}/0"),
                emb.slice(j * PIECE..(j + 1) * PIECE),
            )
        }));
        let state = Arc::new(PeerServeState::new(4, None));
        state.publish_with_memory(vec![], memory).unwrap();
        let sender = Arc::new(Sender::new(12).with_peer_state(state).manifest_only());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let mut w = vec![0u8; dense.len()];
        let meta = download_dense_weight(
            0,
            &url,
            &mut [TensorBuf {
                key: "params.w".into(),
                buf: &mut w,
            }],
            None,
            None,
            Some(PREFIX),
        )
        .await
        .unwrap();
        assert_eq!(&dense[..], &w[..]);
        assert_eq!(
            vec![DenseEntry {
                key: "params.w".into(),
                name: format!("{PREFIX}params.w/c/0"),
                adler32: adler32(&&dense[..]),
            }],
            meta.entries
        );
        assert_eq!(7.0, meta.created_timestamp);

        let mut buf = vec![0u8; emb.len()];
        let (checksum, layout) = download_embedding_table_with_layout(
            &format!("/dev/shm/{PREFIX}"),
            &url,
            "emb",
            &mut buf,
            None,
            None,
            true,
            None,
        )
        .await
        .unwrap();
        assert_eq!(&emb[..], &buf[..]);
        assert_eq!(adler32(&&emb[..]), checksum);
        assert_eq!(
            ShardLayout {
                piece_bytes: PIECE,
                pieces: PIECES
            },
            layout
        );
    }

    async fn spawn_memory_peer() -> (Arc<PeerServeState>, String) {
        spawn_paced_memory_peer(None).await
    }

    async fn spawn_paced_memory_peer(rate: Option<f64>) -> (Arc<PeerServeState>, String) {
        let state = Arc::new(PeerServeState::new(16, rate));
        let sender = Arc::new(
            Sender::new(12)
                .with_peer_state(state.clone())
                .manifest_only(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(CopyService::from_arc(sender))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        (state, url)
    }

    fn publish_pieces(state: &PeerServeState, emb: &Bytes, piece: usize, pieces: &[usize]) {
        let mut memory = vec![(
            format!("{PREFIX}checksums.0.json"),
            Bytes::from_static(b"{}"),
        )];
        memory.extend(pieces.iter().map(|&j| {
            (
                format!("{PREFIX}emb/c/{j}/0"),
                emb.slice(j * piece..(j + 1) * piece),
            )
        }));
        state.publish_with_memory(vec![], memory).unwrap();
    }

    fn table_bytes(seed: usize) -> Bytes {
        Bytes::from(
            (0..PIECE * PIECES)
                .map(|i| ((i * 7 + seed) % 251) as u8)
                .collect::<Vec<u8>>(),
        )
    }

    #[derive(Default)]
    struct Recorded {
        layout: Mutex<Option<ShardLayout>>,
        written: Mutex<Vec<(usize, usize)>>,
    }

    impl DownloadProgress for Recorded {
        fn layout(&self, layout: ShardLayout) {
            *self.layout.lock().unwrap() = Some(layout);
        }

        fn written(&self, start: usize, end: usize) {
            self.written.lock().unwrap().push((start, end));
        }
    }

    async fn progressive(
        urls: &str,
        buf: &mut [u8],
        progress: Option<Arc<dyn DownloadProgress>>,
        stall: Duration,
    ) -> Result<(u32, ShardLayout), CopyPortError> {
        download_embedding_table_progressive(
            &format!("/dev/shm/{PREFIX}"),
            urls,
            "emb",
            buf,
            None,
            Some(4),
            progress,
            stall,
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn progressive_download_follows_a_growing_manifest() {
        let emb = table_bytes(1);
        let (a, url_a) = spawn_memory_peer().await;
        let (b, url_b) = spawn_memory_peer().await;
        publish_pieces(&a, &emb, PIECE, &[]);
        publish_pieces(&b, &emb, PIECE, &[]);
        let grow = {
            let emb = emb.clone();
            tokio::spawn(async move {
                for j in 0..PIECES - 1 {
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    let state = if j % 2 == 0 { &a } else { &b };
                    let listed: Vec<usize> = (0..=j).filter(|i| i % 2 == j % 2).collect();
                    publish_pieces(state, &emb, PIECE, &listed);
                }
                tokio::time::sleep(Duration::from_millis(300)).await;
                a.unpublish();
                tokio::time::sleep(Duration::from_millis(150)).await;
                publish_pieces(&b, &emb, PIECE, &(0..PIECES).collect::<Vec<_>>());
            })
        };
        let recorded = Arc::new(Recorded::default());
        let mut buf = vec![0u8; emb.len()];
        let (checksum, layout) = progressive(
            &format!("{url_a},{url_b}"),
            &mut buf,
            Some(recorded.clone()),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        grow.await.unwrap();

        assert_eq!(&emb[..], &buf[..]);
        assert_eq!(adler32(&&emb[..]), checksum);
        let want = ShardLayout {
            piece_bytes: PIECE,
            pieces: PIECES,
        };
        assert_eq!(
            (want, Some(want)),
            (layout, *recorded.layout.lock().unwrap())
        );
        let mut written = recorded.written.lock().unwrap().clone();
        written.sort();
        assert_eq!(
            (0..PIECES)
                .map(|j| (j * PIECE, (j + 1) * PIECE))
                .collect::<Vec<_>>(),
            written,
            "each piece reported once, whole"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn progressive_download_reads_finer_pieces_out_of_coarser_ones() {
        let emb = table_bytes(2);
        let (fine, url_fine) = spawn_memory_peer().await;
        let (coarse, url_coarse) = spawn_memory_peer().await;
        let fine_pieces: Vec<usize> = (0..PIECES).collect();
        publish_pieces(&fine, &emb, PIECE, &fine_pieces[..PIECES - 1]);
        publish_pieces(&coarse, &emb, PIECE * PIECES, &[]);
        let grow = {
            let emb = emb.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                publish_pieces(&coarse, &emb, PIECE * PIECES, &[0]);
            })
        };

        let mut buf = vec![0u8; emb.len()];
        let (checksum, layout) = progressive(
            &format!("{url_fine},{url_coarse}"),
            &mut buf,
            None,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        grow.await.unwrap();
        assert_eq!(&emb[..], &buf[..]);
        assert_eq!(adler32(&&emb[..]), checksum);
        assert_eq!(PIECE, layout.piece_bytes);
    }

    async fn spawn_freezable_proxy(target: String) -> (Arc<std::sync::atomic::AtomicBool>, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let frozen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        let pump = |mut from: tokio::net::tcp::OwnedReadHalf,
                    mut to: tokio::net::tcp::OwnedWriteHalf,
                    frozen: Arc<std::sync::atomic::AtomicBool>| async move {
            let mut buf = vec![0u8; 64 << 10];
            while let Ok(n @ 1..) = from.read(&mut buf).await {
                while frozen.load(std::sync::atomic::Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                if to.write_all(&buf[..n]).await.is_err() {
                    return;
                }
            }
        };
        let flag = frozen.clone();
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                let server = tokio::net::TcpStream::connect(&target).await.unwrap();
                let ((client_read, client_write), (server_read, server_write)) =
                    (client.into_split(), server.into_split());
                tokio::spawn(pump(client_read, server_write, flag.clone()));
                tokio::spawn(pump(server_read, client_write, flag.clone()));
            }
        });
        (frozen, url)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn progressive_download_is_not_held_up_by_a_hung_replica() {
        let emb = table_bytes(4);
        let (good, url_good) = spawn_memory_peer().await;
        let (hung, url_hung) = spawn_memory_peer().await;
        let (frozen, url_hung) = spawn_freezable_proxy(url_hung).await;
        publish_pieces(&good, &emb, PIECE, &[]);
        publish_pieces(&hung, &emb, PIECE, &[]);
        let grow = {
            let emb = emb.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                frozen.store(true, std::sync::atomic::Ordering::Relaxed);
                for j in 0..PIECES {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    publish_pieces(&good, &emb, PIECE, &(0..=j).collect::<Vec<_>>());
                }
            })
        };
        let mut buf = vec![0u8; emb.len()];
        let t0 = Instant::now();
        let (checksum, _) = progressive(
            &format!("{url_good},{url_hung}"),
            &mut buf,
            None,
            Duration::from_secs(10),
        )
        .await
        .unwrap();
        grow.await.unwrap();
        assert_eq!(&emb[..], &buf[..]);
        assert_eq!(adler32(&&emb[..]), checksum);
        assert!(
            t0.elapsed() < PROGRESSIVE_LIST_TIMEOUT,
            "waited on the hung replica's listings: {:?}",
            t0.elapsed()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn progressive_download_fetches_again_a_piece_whose_send_hangs() {
        let emb = table_bytes(5);
        let (slow, url_slow) = spawn_paced_memory_peer(Some(1.0)).await;
        let (good, url_good) = spawn_memory_peer().await;
        publish_pieces(&slow, &emb, PIECE, &(0..PIECES).collect::<Vec<_>>());
        publish_pieces(&good, &emb, PIECE, &[]);
        let grow = {
            let emb = emb.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(300)).await;
                publish_pieces(&good, &emb, PIECE, &(0..PIECES).collect::<Vec<_>>());
            })
        };
        let mut buf = vec![0u8; emb.len()];
        let stall = Duration::from_secs(4);
        let t0 = Instant::now();
        let (checksum, _) = progressive(&format!("{url_slow},{url_good}"), &mut buf, None, stall)
            .await
            .unwrap();
        grow.await.unwrap();
        assert_eq!(&emb[..], &buf[..]);
        assert_eq!(adler32(&&emb[..]), checksum);
        assert!(t0.elapsed() < stall, "{:?}", t0.elapsed());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn progressive_download_fails_once_no_piece_arrives() {
        let emb = table_bytes(3);
        let (stuck, url) = spawn_memory_peer().await;
        publish_pieces(&stuck, &emb, PIECE, &[0, 1]);

        let mut buf = vec![0u8; emb.len()];
        let t0 = Instant::now();
        let err = progressive(&url, &mut buf, None, Duration::from_millis(500))
            .await
            .unwrap_err();
        assert!(matches!(err, CopyPortError::Timeout(_)), "{err}");
        assert!(err.to_string().contains("timeout"), "{err}");
        assert!(t0.elapsed() >= Duration::from_millis(500));
        assert_eq!(&emb[..2 * PIECE], &buf[..2 * PIECE]);
    }
}
