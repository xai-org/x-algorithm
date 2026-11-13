// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::{
    checkpoint_proxy_metrics::*,
    r#const::ROOT_DIR,
    emb_table::{apply_copy_port_http2, list_entries, send_entries},
};
use log::{info, warn};
use memmap2::MmapMut;
use std::{
    cmp,
    collections::BTreeMap,
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tonic::transport;

struct CheckpointPaths {
    grpc_prefix: String,
    staging_top: PathBuf,
    staging_dir: PathBuf,
    final_dir: PathBuf,
}

impl CheckpointPaths {
    fn new(prefix: &str) -> Self {
        let top_dir = prefix.split('/').next().unwrap_or(prefix);
        Self {
            grpc_prefix: format!("{}/", prefix),
            staging_top: PathBuf::from(format!("{}.{}", ROOT_DIR, top_dir)),
            staging_dir: PathBuf::from(format!("{}.{}", ROOT_DIR, prefix)),
            final_dir: PathBuf::from(format!("{}{}", ROOT_DIR, top_dir)),
        }
    }

    fn to_local<'a>(&self, grpc_name: &'a str) -> &'a str {
        grpc_name
            .strip_prefix(&self.grpc_prefix)
            .unwrap_or(grpc_name)
    }
}

const DEFAULT_CHUNK_SIZE: usize = 256 << 20;

pub struct CheckpointProxy {
    copy_url: String,
    server_index: usize,
    num_servers: usize,
    last_prefix: tokio::sync::Mutex<String>,
    downloading: AtomicBool,
    max_entries: usize,
    verify_checksums: bool,
    num_trainers: Option<usize>,
    download_concurrency: usize,
    request_timeout: Duration,
}

impl CheckpointProxy {
    pub fn new(
        copy_url: String,
        server_index: usize,
        num_servers: usize,
        num_trainers: Option<usize>,
        max_entries: usize,
        verify_checksums: bool,
        download_concurrency: usize,
        request_timeout: Duration,
    ) -> Self {
        Self {
            copy_url,
            server_index,
            num_servers,
            num_trainers,
            last_prefix: tokio::sync::Mutex::new(String::new()),
            downloading: AtomicBool::new(false),
            max_entries,
            verify_checksums,
            download_concurrency,
            request_timeout,
        }
    }

    async fn resolve_and_connect(
        &self,
    ) -> Result<Vec<(SocketAddr, transport::Channel)>, io::Error> {
        let (scheme, hostport) = if let Some(rest) = self.copy_url.strip_prefix("https://") {
            ("https", rest)
        } else {
            ("http", self.copy_url.trim_start_matches("http://"))
        };
        let tls = crate::tls::ClientTlsOptions::from_env()?;
        let (name, port) = hostport.rsplit_once(':').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("copy_url must be host:port, got {}", self.copy_url),
            )
        })?;

        let connect_timeout = std::time::Duration::from_secs(
            std::env::var("COPY_PORT_CONNECT_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        );

        let lookup = format!("{}:{}", name, port);
        let addrs: Vec<SocketAddr> =
            tokio::time::timeout(connect_timeout, tokio::net::lookup_host(&lookup))
                .await
                .map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!("DNS lookup for {lookup} timed out after {connect_timeout:?}"),
                    )
                })??
                .collect();

        if addrs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("DNS resolution returned 0 addresses for {}", name),
            ));
        }

        let dns_count = addrs.len();

        let request_timeout = self.request_timeout;

        let mut endpoints = Vec::with_capacity(addrs.len());
        for addr in &addrs {
            endpoints.push(crate::tls::endpoint_for(
                &format!("{scheme}://{addr}"),
                tls.as_ref(),
            )?);
        }
        let results =
            futures::future::join_all(addrs.iter().zip(endpoints).map(|(addr, endpoint)| {
                let timeout = connect_timeout;
                async move {
                    let result = apply_copy_port_http2(endpoint)
                        .connect_timeout(timeout)
                        .timeout(request_timeout)
                        .http2_keep_alive_interval(Duration::from_secs(30))
                        .keep_alive_timeout(Duration::from_secs(10))
                        .keep_alive_while_idle(true)
                        .connect()
                        .await;
                    (*addr, result)
                }
            }))
            .await;

        let mut connected: Vec<(SocketAddr, transport::Channel)> = results
            .into_iter()
            .filter_map(|(addr, result)| match result {
                Ok(channel) => Some((addr, channel)),
                Err(e) => {
                    info!("connect failed for {} (filtered out): {}", addr, e);
                    None
                }
            })
            .collect();

        if connected.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "failed to connect to any trainer",
            ));
        }

        connected.sort_by_key(|(addr, _)| *addr);

        if connected.len() != dns_count {
            info!(
                "DNS returned {} addresses, {} reachable (unreachable nodes filtered out)",
                dns_count,
                connected.len(),
            );
        }

        assign_trainer_channels(
            connected,
            self.num_trainers,
            self.server_index,
            self.num_servers,
        )
    }

    async fn extra_trainer_channels(&self, addr: SocketAddr, n: usize) -> Vec<transport::Channel> {
        if n == 0 {
            return Vec::new();
        }
        let scheme = if self.copy_url.starts_with("https://") {
            "https"
        } else {
            "http"
        };
        let tls = match crate::tls::ClientTlsOptions::from_env() {
            Ok(tls) => tls,
            Err(e) => {
                warn!("extra trainer channels: tls config: {e}");
                return Vec::new();
            }
        };
        let connect_timeout = Duration::from_secs(
            std::env::var("COPY_PORT_CONNECT_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        );
        let request_timeout = self.request_timeout;
        let url = format!("{scheme}://{addr}");
        let results = futures::future::join_all((0..n).map(|_| {
            let tls = tls.as_ref();
            let url = url.clone();
            async move {
                let endpoint = crate::tls::endpoint_for(&url, tls).ok()?;
                apply_copy_port_http2(endpoint)
                    .connect_timeout(connect_timeout)
                    .timeout(request_timeout)
                    .http2_keep_alive_interval(Duration::from_secs(30))
                    .keep_alive_timeout(Duration::from_secs(10))
                    .keep_alive_while_idle(true)
                    .connect()
                    .await
                    .ok()
            }
        }))
        .await;
        results.into_iter().flatten().collect()
    }

    pub async fn poll_and_download(
        &self,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        POLL_TOTAL.inc();

        if self
            .downloading
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            info!("poll: download already in progress, skipping");
            return Ok(false);
        }
        let _guard = scopeguard::guard((), |_| {
            self.downloading.store(false, Ordering::SeqCst);
        });

        let assigned = self.resolve_and_connect().await?;
        TRAINER_URL_COUNT.set(assigned.len() as f64);
        let channels: Vec<transport::Channel> = assigned.iter().map(|(_, ch)| ch.clone()).collect();

        let entries = list_entries(&channels, "").await?;

        let latest_prefix = find_latest_complete_prefix(&entries, channels.len());
        let Some(prefix) = latest_prefix else {
            info!("poll: no complete checkpoint found on all trainers");
            return Ok(false);
        };

        {
            let last = self.last_prefix.lock().await;
            if !last.is_empty() && prefix <= *last {
                return Ok(false);
            }
        }

        info!("poll: found new checkpoint {}", prefix);
        let t0 = Instant::now();

        let paths = CheckpointPaths::new(&prefix);
        let mut checkpoint_entries: Vec<Vec<(String, usize)>> = Vec::with_capacity(entries.len());
        for channel_entries in &entries {
            let filtered: Vec<(String, usize)> = channel_entries
                .iter()
                .filter(|(name, _)| name.starts_with(&paths.grpc_prefix))
                .cloned()
                .collect();
            checkpoint_entries.push(filtered);
        }

        match self
            .download_checkpoint(&paths, &assigned, &checkpoint_entries)
            .await
        {
            Ok(()) => {}
            Err(e) => {
                DOWNLOAD_ERRORS_TOTAL.inc();
                return Err(e);
            }
        }

        self.finalize_and_cleanup(&paths)?;

        let elapsed = t0.elapsed();
        let total_bytes: usize = checkpoint_entries
            .iter()
            .flat_map(|v| v.iter())
            .map(|(_, s)| s)
            .sum();

        DOWNLOAD_TOTAL.inc();
        DOWNLOAD_DURATION_SECONDS.observe(elapsed.as_secs_f64());
        DOWNLOAD_BYTES_TOTAL.inc_by(total_bytes as f64);
        if elapsed.as_secs_f64() > 0.0 {
            DOWNLOAD_RATE_BYTES_PER_SEC.observe(total_bytes as f64 / elapsed.as_secs_f64());
        }
        LATEST_CHECKPOINT_TIMESTAMP.set(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        );

        info!(
            "checkpoint {} downloaded and finalized: {:.2} GB in {:.1}s ({:.2} GB/s)",
            prefix,
            total_bytes as f64 * 1e-9,
            elapsed.as_secs_f64(),
            total_bytes as f64 * 1e-9 / elapsed.as_secs_f64(),
        );

        *self.last_prefix.lock().await = prefix;
        Ok(true)
    }

    async fn download_checkpoint(
        &self,
        paths: &CheckpointPaths,
        trainers: &[(SocketAddr, transport::Channel)],
        entries: &[Vec<(String, usize)>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let concurrency = self.download_concurrency.max(1);

        let mut seen_files = std::collections::HashSet::new();
        let mut deduped_entries: Vec<Vec<(String, usize)>> =
            entries.iter().map(|_| Vec::new()).collect();

        for (ch_idx, ch_entries) in entries.iter().enumerate() {
            for (name, size) in ch_entries {
                let local_name = paths.to_local(name);
                if seen_files.insert(local_name.to_string()) {
                    deduped_entries[ch_idx].push((name.clone(), *size));
                }
            }
        }

        let required: u64 = deduped_entries
            .iter()
            .flat_map(|v| v.iter())
            .map(|(_, s)| *s as u64)
            .sum();
        self.ensure_free_space(required)?;

        if paths.staging_top.exists() {
            fs::remove_dir_all(&paths.staging_top)?;
        }
        fs::create_dir_all(&paths.staging_dir)?;

        let total_deduped: usize = deduped_entries.iter().map(|e| e.len()).sum();
        let total_original: usize = entries.iter().map(|e| e.len()).sum();
        if total_deduped < total_original {
            info!(
                "deduplicated {total_original} -> {total_deduped} entries ({} replicated files skipped)",
                total_original - total_deduped,
            );
        }

        let mut handles = Vec::with_capacity(trainers.len());

        for (ch_idx, ((addr, channel), ch_entries)) in
            trainers.iter().zip(deduped_entries).enumerate()
        {
            if ch_entries.is_empty() {
                continue;
            }

            let extras = concurrency.saturating_sub(1);
            let mut pool = vec![channel.clone()];
            if extras > 0 {
                let extra = self.extra_trainer_channels(*addr, extras).await;
                if extra.len() < extras {
                    warn!(
                        "trainer {addr}: opened {}/{} extra connections; downloading on {}",
                        extra.len(),
                        extras,
                        extra.len() + 1
                    );
                }
                pool.extend(extra);
            }
            info!(
                "trainer {addr}: downloading on {} HTTP/2 connection(s) (concurrency={concurrency})",
                pool.len(),
            );

            let staging_dir = paths.staging_dir.clone();
            let grpc_prefix = paths.grpc_prefix.clone();
            let verify = self.verify_checksums;

            let handle = tokio::spawn(async move {
                download_channel_files(
                    &staging_dir,
                    &grpc_prefix,
                    pool,
                    &ch_entries,
                    ch_idx,
                    verify,
                )
                .await
            });
            handles.push(handle);
        }

        let results: Vec<_> = futures::future::join_all(handles).await;
        for result in results {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    let _ = fs::remove_dir_all(&paths.staging_top);
                    return Err(e);
                }
                Err(e) => {
                    let _ = fs::remove_dir_all(&paths.staging_top);
                    return Err(Box::new(e));
                }
            }
        }

        Ok(())
    }

    fn finalize_and_cleanup(&self, paths: &CheckpointPaths) -> io::Result<()> {
        fs::rename(&paths.staging_top, &paths.final_dir)?;
        info!(
            "finalized: {} -> {}",
            paths.staging_top.display(),
            paths.final_dir.display(),
        );
        self.cleanup_old_checkpoints();
        Ok(())
    }

    fn cleanup_old_checkpoints(&self) {
        let Ok(read_dir) = fs::read_dir(ROOT_DIR) else {
            return;
        };

        let mut elapsed_dirs: Vec<String> = read_dir
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name().to_str()?.to_string();
                if name.starts_with("elapsed_samples_") && entry.path().is_dir() {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();

        elapsed_dirs.sort();
        CHECKPOINTS_AVAILABLE.set(elapsed_dirs.len() as f64);

        if elapsed_dirs.len() > self.max_entries {
            let to_remove = elapsed_dirs.len() - self.max_entries;
            for name in &elapsed_dirs[..to_remove] {
                let path = format!("{}{}", ROOT_DIR, name);
                info!("cleanup: removing old checkpoint {}", path);
                if let Err(e) = fs::remove_dir_all(&path) {
                    warn!("cleanup: failed to remove {}: {}", path, e);
                }
            }
        }

        cleanup_stale_staging_dirs();
    }

    fn ensure_free_space(&self, required: u64) -> io::Result<()> {
        let needed = required + required / 20;

        cleanup_stale_staging_dirs();
        let mut free = free_bytes(ROOT_DIR)?;
        if free >= needed {
            return Ok(());
        }

        let read_dir = fs::read_dir(ROOT_DIR).map_err(|e| {
            io::Error::new(
                io::ErrorKind::StorageFull,
                format!(
                    "need {:.1} GiB but only {:.1} GiB free, and cannot enumerate {} to evict: {}",
                    needed as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64,
                    ROOT_DIR,
                    e,
                ),
            )
        })?;
        let mut elapsed_dirs: Vec<String> = read_dir
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name().to_str()?.to_string();
                (name.starts_with("elapsed_samples_") && entry.path().is_dir()).then_some(name)
            })
            .collect();
        elapsed_dirs.sort();

        while free < needed && elapsed_dirs.len() > 1 {
            let name = elapsed_dirs.remove(0);
            let path = format!("{}{}", ROOT_DIR, name);
            warn!(
                "low space: need {:.1} GiB, {:.1} GiB free — evicting old checkpoint {}",
                needed as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64,
                path,
            );
            if let Err(e) = fs::remove_dir_all(&path) {
                warn!("eviction failed for {}: {}", path, e);
                break;
            }
            free = free_bytes(ROOT_DIR)?;
        }

        if free < needed {
            return Err(io::Error::new(
                io::ErrorKind::StorageFull,
                format!(
                    "not enough space for checkpoint: need {:.1} GiB, {:.1} GiB free after eviction",
                    needed as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64,
                ),
            ));
        }
        Ok(())
    }

    pub fn has_existing_checkpoints(&self) -> bool {
        let Ok(read_dir) = fs::read_dir(ROOT_DIR) else {
            return false;
        };
        read_dir.filter_map(|e| e.ok()).any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("elapsed_samples_"))
                && e.path().is_dir()
        })
    }
}

async fn download_channel_files(
    staging_dir: &Path,
    grpc_prefix: &str,
    channels: Vec<transport::Channel>,
    entries: &[(String, usize)],
    ch_idx: usize,
    _verify_checksums: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let t0 = Instant::now();
    let mut total_bytes: usize = 0;

    for (name, size) in entries {
        let local_name = name.strip_prefix(grpc_prefix).unwrap_or(name);
        let file_path = staging_dir.join(local_name);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if *size == 0 {
            fs::File::create(&file_path)?;
        }
    }

    let n_conns = channels.len().max(1);
    let sem = Arc::new(tokio::sync::Semaphore::new(n_conns));
    let mut handles = Vec::new();
    let mut chunk_idx = 0usize;

    for (name, size) in entries {
        if *size == 0 {
            continue;
        }

        let local_name = name.strip_prefix(grpc_prefix).unwrap_or(name);
        let file_path = staging_dir.join(local_name);

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&file_path)
            .map_err(|e| format!("failed to create {}: {}", file_path.display(), e))?;
        file.set_len(*size as u64).map_err(|e| {
            format!(
                "failed to set_len({}) on {}: {}",
                size,
                file_path.display(),
                e
            )
        })?;
        let mmap_inner = unsafe { MmapMut::map_mut(&file) }
            .map_err(|e| format!("failed to mmap {}: {}", file_path.display(), e))?;
        if mmap_inner.len() != *size {
            return Err(format!(
                "mmap size mismatch for {}: expected {}, got {} (name={}, local_name={})",
                file_path.display(),
                size,
                mmap_inner.len(),
                name,
                local_name,
            )
            .into());
        }
        let mmap = Arc::new(parking_lot::Mutex::new(mmap_inner));

        let chunk_size = DEFAULT_CHUNK_SIZE;
        let mut offset = 0usize;
        while offset < *size {
            let this_chunk = cmp::min(chunk_size, *size - offset);
            let channel = channels[chunk_idx % n_conns].clone();
            chunk_idx += 1;
            let name = name.clone();
            let sem = sem.clone();
            let mmap = mmap.clone();
            let chunk_offset = offset;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                download_chunk(channel, &name, chunk_offset, this_chunk, &mmap, ch_idx).await
            });
            handles.push(handle);
            offset += this_chunk;
        }

        total_bytes += size;
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    for result in results {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(Box::new(e)),
        }
    }

    let elapsed = t0.elapsed();
    CHANNEL_DOWNLOAD_DURATION_SECONDS.observe(elapsed.as_secs_f64());
    if elapsed.as_secs_f64() > 0.0 {
        CHANNEL_DOWNLOAD_RATE_BYTES_PER_SEC.observe(total_bytes as f64 / elapsed.as_secs_f64());
    }
    info!(
        "channel {}: downloaded {} files, {:.2} GB in {:.1}s ({:.2} GB/s) [connections={}]",
        ch_idx,
        entries.len(),
        total_bytes as f64 * 1e-9,
        elapsed.as_secs_f64(),
        if elapsed.as_secs_f64() > 0.0 {
            total_bytes as f64 * 1e-9 / elapsed.as_secs_f64()
        } else {
            0.0
        },
        n_conns,
    );

    Ok(())
}

async fn download_chunk(
    channel: transport::Channel,
    name: &str,
    offset: usize,
    size: usize,
    mmap: &Arc<parking_lot::Mutex<MmapMut>>,
    ch_idx: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let padded = size + 64;
    let mut buf = vec![0u8; padded];
    let buf_ref: &'static mut [u8] = unsafe { std::mem::transmute(&mut buf[..]) };

    let (bytes_written, _checksum) = send_entries(
        channel,
        vec![name.as_bytes().to_vec()],
        vec![offset],
        vec![size],
        buf_ref,
        String::new(),
        #[cfg(target_os = "linux")]
        (Vec::new(), Arc::new(Vec::new()), Arc::new(Vec::new())),
    )
    .await;

    if bytes_written == usize::MAX {
        return Err(format!(
            "channel {}: gRPC transfer failed for {} (offset={}, size={})",
            ch_idx, name, offset, size,
        )
        .into());
    }
    if bytes_written != size {
        return Err(format!(
            "channel {}: size mismatch for {} offset={}: expected {}, got {}",
            ch_idx, name, offset, size, bytes_written,
        )
        .into());
    }

    {
        let mut mmap_guard = mmap.lock();
        mmap_guard[offset..offset + size].copy_from_slice(&buf[..size]);
    }

    Ok(())
}

fn assign_trainer_channels(
    connected: Vec<(SocketAddr, transport::Channel)>,
    num_trainers: Option<usize>,
    server_index: usize,
    num_servers: usize,
) -> Result<Vec<(SocketAddr, transport::Channel)>, io::Error> {
    let total_trainers = if let Some(expected) = num_trainers {
        if connected.len() < expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "only {} trainers reachable but --num-trainers expects {}",
                    connected.len(),
                    expected,
                ),
            ));
        }
        expected
    } else {
        connected.len()
    };

    if !total_trainers.is_multiple_of(num_servers) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "connected trainer count {} is not divisible by num_servers {}",
                total_trainers, num_servers,
            ),
        ));
    }

    let per_server = total_trainers / num_servers;
    let start = server_index * per_server;

    let my_slice: Vec<_> = connected
        .into_iter()
        .take(total_trainers)
        .skip(start)
        .take(per_server)
        .collect();

    let my_ips: Vec<_> = my_slice.iter().map(|(addr, _)| addr.to_string()).collect();

    info!(
        "{} trainers reachable (sorted by IP), this server (index {}) mirrors {}-{}: {:?}",
        total_trainers,
        server_index,
        start,
        start + per_server - 1,
        my_ips,
    );

    Ok(my_slice)
}

fn free_bytes(path: &str) -> io::Result<u64> {
    let stat = nix::sys::statvfs::statvfs(path)
        .map_err(|e| io::Error::other(format!("statvfs({}): {}", path, e)))?;
    #[allow(clippy::unnecessary_cast)]
    let free = stat.blocks_available() as u64 * stat.fragment_size() as u64;
    Ok(free)
}

pub fn cleanup_stale_staging_dirs() {
    let Ok(read_dir) = fs::read_dir(ROOT_DIR) else {
        return;
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(".elapsed_samples_") && entry.path().is_dir() {
            let path = entry.path();
            info!("cleanup: removing stale staging dir {}", path.display());
            let _ = fs::remove_dir_all(&path);
        }
    }
}

fn extract_prefix(name: &str) -> Option<&str> {
    let i = name.find('/')?;
    let j = name[i + 1..].find('/')?;
    Some(&name[..i + j + 1])
}

fn find_latest_complete_prefix(
    entries: &[Vec<(String, usize)>],
    num_channels: usize,
) -> Option<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for channel_entries in entries {
        let mut seen_prefixes = std::collections::HashSet::new();
        for (name, _) in channel_entries {
            if let Some(prefix) = extract_prefix(name)
                && seen_prefixes.insert(prefix.to_string())
            {
                *counts.entry(prefix.to_string()).or_default() += 1;
            }
        }
    }

    let mut latest: Option<String> = None;
    for (prefix, count) in &counts {
        if *count == num_channels {
            latest = Some(prefix.clone());
        }
    }
    latest
}
