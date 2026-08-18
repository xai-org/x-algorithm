// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::{
    r#const::{LIST, REGISTER, ROOT_DIR, SEND},
    copy_manifest::{PacedBody, PeerManifest, PeerServeState},
    grpc_util::{add_ok_trailer, concat_bodies, freeze, get_bytes_mut, make_response},
    ibverbs_util::{MAX_QUEUE_DEPTH, MAX_WR_SIZE, handshake, make_ib_contexts, make_qp, poll},
    proto,
    proto_parser::{
        BodyKind, BytesMutProtoExt, bytes, decode_names, encode_names, parse, proto,
        repeated_bytes, repeated_ints,
    },
};
use bytes::{Bytes, BytesMut};
use futures::future::{BoxFuture, join_all};
use http_body_util::Full;
use memmap2;
use std::{
    cmp,
    collections::BTreeMap,
    convert::Infallible,
    fs, io, mem,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    str,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    sync::{Mutex, MutexGuard},
    task,
};
use tonic::{
    Status, body,
    codegen::{Service, http},
    server::NamedService,
};

pub use crate::copy_manifest::ManifestSpec;

pub type Contexts = Arc<Vec<(ibverbs::Context, ibverbs::ProtectionDomain)>>;
pub type MR = ibverbs::MemoryRegion<&'static mut [u8]>;
pub type MRs = Arc<Vec<Option<MR>>>;
pub type MRz = Arc<Vec<Vec<MR>>>;

#[allow(clippy::ptr_arg)]
pub fn get_rmr(rmr: &Vec<u8>) -> Result<ibverbs::RemoteMemorySlice, Status> {
    bincode::deserialize(rmr).map_err(|e| Status::from_error(Box::new(e)))
}

pub async fn register_memory_regions(
    sizes: &[usize],
    devices: &[usize],
    contexts: Contexts,
    buf: &mut [u8],
    offset: &mut usize,
) -> Result<Vec<Vec<MR>>, Status> {
    let mut mrz = Vec::with_capacity(sizes.len());
    for &size in sizes {
        if devices.is_empty() {
            break;
        }
        let step = size.div_ceil(devices.len());
        let min_size = 2 * devices.len() * devices.len();
        mrz.push(
            join_all(devices.iter().enumerate().map(|(i, &j)| {
                let contexts = contexts.clone();
                let offset = *offset;
                let buf: &'static mut [u8] = unsafe { std::mem::transmute(&mut buf[..]) };
                task::spawn(async move {
                    let a = i * step;
                    let b = cmp::min(size, a + step);
                    if j >= contexts.len() || a == b {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("index={j}/{}, length={}, size={size} should be greater than {min_size}", contexts.len(), b - a),
                        ))?;
                    }
                    contexts[j].1.register(&mut buf[offset + a..offset + b])
                })
            }))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Status::from_error(Box::new(e)))?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?,
        );
        *offset += size;
    }
    Ok(mrz)
}

pub fn make_queues(
    is_read: bool,
    devices: &[usize],
    contexts: &[(ibverbs::Context, ibverbs::ProtectionDomain)],
    mrz: &[Vec<MR>],
    bytes: &mut BytesMut,
) -> Result<Vec<(ibverbs::PreparedQueuePair, ibverbs::CompletionQueue)>, Status> {
    let mut queues = Vec::with_capacity(devices.len());
    for (i, (ctx, pd)) in contexts.iter().enumerate() {
        if devices.iter().all(|&j| i != j) {
            bytes.put_string(6, &b""[..]);
            continue;
        }
        let (ep, qp, cq) = make_qp(is_read, ctx, pd, MAX_QUEUE_DEPTH, 0)?;
        bytes.put_string(6, &ep);
        queues.push((qp, cq));
    }
    for mrs in mrz.iter().take(if is_read { 0 } else { mrz.len() }) {
        for mr in mrs {
            let mr =
                bincode::serialize(&mr.remote()).map_err(|e| Status::from_error(Box::new(e)))?;
            bytes.put_string(7, &mr);
        }
    }
    Ok(queues)
}

pub async fn rdma(
    is_read: bool,
    sizes: Vec<usize>,
    devices: Vec<usize>,
    contexts: Contexts,
    mrz: MRz,
    queues: Vec<(ibverbs::PreparedQueuePair, ibverbs::CompletionQueue)>,
    endpoints: Vec<Vec<u8>>,
    rmrs: Vec<Vec<u8>>,
    use_rdma: Vec<u8>,
) -> Result<(), Status> {
    if endpoints.len() != queues.len() {
        return Err(Status::internal(format!(
            "bad rdma_endpoints count: wanted {}, got {}",
            queues.len(),
            endpoints.len()
        )));
    }
    let num_rdma = use_rdma.iter().fold(0, |acc, &x| acc + x as usize);
    let num_rmrs = if is_read { queues.len() * num_rdma } else { 0 };
    if rmrs.len() != num_rmrs {
        return Err(Status::internal(format!(
            "bad rdma_memory_regions count: wanted {}, got {}",
            num_rmrs,
            rmrs.len()
        )));
    }
    if num_rdma == 0 {
        return Ok(());
    }

    let mut queues = handshake(endpoints, queues)?;
    let mut count = 0;
    let mut j = 0;
    let t = std::time::Instant::now();
    for (i, (&size, &use_rdma)) in sizes.iter().zip(&use_rdma).enumerate() {
        if use_rdma == 0 || !is_read {
            continue;
        }

        let mrs = &mrz[i];
        let step = size.div_ceil(queues.len());
        let rmrs = rmrs.iter().skip(j * queues.len()).map(get_rmr);
        for (i, (((qp, _), mr), rmr)) in queues.iter_mut().zip(mrs).zip(rmrs).enumerate() {
            let rmr = rmr?;
            let offset = i * step;
            let limit = cmp::min(size - offset, step);
            assert_eq!(limit, mr.remote().len);
            for a in (0..limit).step_by(MAX_WR_SIZE) {
                let b = cmp::min(a + MAX_WR_SIZE, limit);
                let (x, y) = (offset + a, offset + b);
                qp.post_read(&[mr.slice(a..b)], rmr.slice(x..y), count as u64)?;
                count += 1;
            }
        }
        j += 1;
    }
    if !is_read {
        let mrs = devices.iter().map(|&i| contexts[i].1.allocate(16));
        for ((qp, _), mr) in queues.iter_mut().zip(mrs) {
            unsafe { qp.post_receive(&[mr?.slice(..)], 0)? };
        }
        count = queues.len();
    }
    poll(count, &queues).await?;

    let f = |acc, (x, y)| if y != 0 { acc + x } else { acc };
    let size = sizes.into_iter().zip(use_rdma).fold(0, f);
    let x = size as f64 / t.elapsed().as_secs_f64() * 1e-9;
    log::info!("RDMA: {:.2} GB/s", x);
    Ok(())
}

async fn handle_rdma(
    endpoints: Vec<(usize, Vec<u8>)>,
    queues: Vec<(ibverbs::PreparedQueuePair, ibverbs::CompletionQueue)>,
    mrz: Vec<(MRs, usize, usize)>,
    rmrs: Vec<Vec<u8>>,
) -> Result<(), Status> {
    let devices: Vec<_> = endpoints.iter().map(|&(i, _)| i).collect();
    let mut queues = handshake(endpoints.into_iter().map(|(_, x)| x).collect(), queues)?;
    if rmrs.is_empty() {
        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
        return Ok(());
    }
    let t = std::time::Instant::now();
    let mut count = 0;
    for (j, &(ref mrs, offset, size)) in mrz.iter().enumerate() {
        let step = size.div_ceil(queues.len());
        let mrs = devices.iter().map(|&i| mrs[i].as_ref().unwrap());
        let rmrs = rmrs.iter().skip(j * queues.len()).map(get_rmr);
        for (i, (((qp, _), mr), rmr)) in queues.iter_mut().zip(mrs).zip(rmrs).enumerate() {
            let rmr = rmr?;
            let limit = cmp::min(size - i * step, step);
            let offset = offset + i * step;
            assert_eq!(limit, rmr.len);
            for a in (0..limit).step_by(MAX_WR_SIZE) {
                let b = cmp::min(a + MAX_WR_SIZE, limit);
                let (x, y) = (offset + a, offset + b);
                let last = b == limit && j == mrz.len() - 1;
                const IMM_DATA: u32 = 0x42;
                let flags = if last { Some(IMM_DATA) } else { None };
                qp.post_write(&[mr.slice(x..y)], rmr.slice(a..b), count as u64, flags)?;
                count += 1;
            }
        }
    }
    poll(count, &queues).await?;
    let size = mrz.iter().fold(0, |acc, x| acc + x.2);
    log::info!("{:.2} GB/s", size as f64 / t.elapsed().as_secs_f64() * 1e-9);
    Ok(())
}

pub struct Sender {
    bytez: Mutex<BTreeMap<String, (Bytes, MRs)>>,
    contexts: Vec<(ibverbs::Context, ibverbs::ProtectionDomain)>,
    max_entries: usize,
    peer_serve: Option<PeerServeState>,
}

pub const DEFAULT_MAX_ENTRIES: usize = 12;

impl Default for Sender {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }
}

impl Sender {
    pub fn new(max_entries: usize) -> Self {
        Self {
            bytez: Mutex::new(BTreeMap::new()),
            contexts: make_ib_contexts().unwrap(),
            max_entries,
            peer_serve: None,
        }
    }

    pub fn with_peer_serve(
        mut self,
        max_concurrent_sends: usize,
        rate_limit_bytes_per_sec: Option<f64>,
    ) -> Self {
        self.peer_serve = Some(PeerServeState::new(
            max_concurrent_sends,
            rate_limit_bytes_per_sec,
        ));
        self
    }

    pub fn publish_manifest(
        &self,
        entries: Vec<ManifestSpec>,
        blobs: Vec<(String, Vec<u8>)>,
    ) -> io::Result<()> {
        let Some(peer_serve) = &self.peer_serve else {
            return Err(io::Error::other("peer serving disabled"));
        };
        peer_serve.publish(entries, blobs)
    }

    pub fn revoke_manifest(&self, drain_timeout: Duration) -> bool {
        let Some(peer_serve) = &self.peer_serve else {
            return true;
        };
        peer_serve.revoke(drain_timeout)
    }

    fn manifest(&self) -> Option<Arc<PeerManifest>> {
        self.peer_serve.as_ref().and_then(|p| p.manifest())
    }
}

fn walk_dir(path: &Path, start: usize, max_entries: usize) -> io::Result<Vec<(Vec<u8>, usize)>> {
    let is_root = path.as_os_str().len() == start;
    let mut entries = fs::read_dir(path)?
        .filter_map(|entry| {
            if let Ok(entry) = entry
                && (!is_root
                    || entry
                        .file_name()
                        .as_os_str()
                        .to_str()?
                        .starts_with("elapsed"))
            {
                Some(entry)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if is_root {
        entries.sort_by_key(|entry| entry.file_name());
    }
    let mut entry_paths = Vec::new();
    for entry in &entries[(if is_root && entries.len() > max_entries {
        entries.len() - max_entries
    } else {
        0
    })..]
    {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            entry_paths.extend(walk_dir(&entry_path, start, max_entries)?);
        } else if !is_root {
            entry_paths.push((
                entry_path.to_string_lossy()[start..]
                    .to_string()
                    .into_bytes(),
                fs::metadata(entry_path)?.len() as usize,
            ));
        }
    }
    Ok(entry_paths)
}

fn mmap_file(
    path: &str,
    name: &str,
    contexts: &[(ibverbs::Context, ibverbs::ProtectionDomain)],
    context_indexes: &[usize],
) -> io::Result<(Bytes, MRs)> {
    task::block_in_place(|| {
        let file = open_checkpoint_file(Path::new(path), Path::new(name))?;
        let mut mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
        let mut mrs: Vec<_> = (0..contexts.len()).map(|_| None).collect();
        for &i in context_indexes {
            let b: &'static mut [u8] = unsafe { mem::transmute(mmap.as_mut()) };
            mrs[i] = Some(contexts[i].1.register(b)?);
        }
        Ok((Bytes::from_owner(mmap), Arc::new(mrs)))
    })
}

fn open_checkpoint_file(path: &Path, name: &Path) -> io::Result<fs::File> {
    open_file_within(Path::new(ROOT_DIR), path, name)
}

fn open_file_within(root: &Path, path: &Path, name: &Path) -> io::Result<fs::File> {
    let root = root.canonicalize()?;
    let mut file = path.as_os_str().to_os_string();
    file.push(name.as_os_str());
    let file = PathBuf::from(file).canonicalize()?;
    if !file.starts_with(&root) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "checkpoint file is outside the checkpoint root",
        ));
    }

    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(false)
        .truncate(false)
        .open(file)?;
    ensure_open_file_within(&root, &file)?;
    Ok(file)
}

fn ensure_open_file_within(root: &Path, file: &fs::File) -> io::Result<()> {
    let opened_path =
        PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd())).canonicalize()?;
    if !opened_path.starts_with(root) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "opened checkpoint file is outside the checkpoint root",
        ));
    }
    Ok(())
}

fn cleanup<'a>(mut bytez: MutexGuard<'a, BTreeMap<String, (Bytes, MRs)>>, max_entries: usize) {
    let mut prev = "";
    let mut splits = Vec::new();
    for (i, (k, _)) in bytez.iter().enumerate() {
        let this = &k[..k.find("/").unwrap_or(k.len())];
        if prev != this {
            prev = this;
            splits.push(i);
        }
    }
    let mut values_to_remove = Vec::new();
    if splits.len() > max_entries {
        let keys_to_remove = bytez
            .iter()
            .take(splits[splits.len() - max_entries])
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>();
        values_to_remove.reserve(keys_to_remove.len());
        for key in keys_to_remove {
            values_to_remove.push(bytez.remove(&key).unwrap());
        }
    }
    drop(bytez);
    if !values_to_remove.is_empty() {
        task::spawn_blocking(move || {
            drop(values_to_remove);
        });
    }
}

impl Sender {
    pub async fn handle_list(self: Arc<Self>, body: body::Body) -> Result<body::Body, Status> {
        parse(proto![], body, BodyKind::Request).await?;

        let mut paths = walk_dir(Path::new(ROOT_DIR), ROOT_DIR.len(), self.max_entries)
            .map_err(|e| Status::from_error(Box::new(e)))?;
        if let Some(manifest) = self.manifest() {
            paths.extend(manifest.list());
        }
        let (names, sizes): (Vec<_>, Vec<_>) = paths.into_iter().unzip();

        let (all_names, prefix_sizes, suffix_sizes) = encode_names(names);
        let mut buf = get_bytes_mut();
        buf.put_string(1, &all_names);
        buf.put_repeated_ints([2, 3, 4], [&prefix_sizes, &suffix_sizes, &sizes]);
        Ok(add_ok_trailer(freeze(buf)))
    }

    pub async fn handle_send(self: Arc<Self>, body: body::Body) -> Result<body::Body, Status> {
        let mut names = Vec::new();
        let mut prefix_sizes = Vec::new();
        let mut suffix_sizes = Vec::new();
        let mut offsets = Vec::<usize>::new();
        let mut sizes = Vec::<usize>::new();
        let mut endpoints = Vec::with_capacity(self.contexts.len());
        let mut rmrs = Vec::new();
        let mut prio = 0usize;
        let proto = proto![
            (1, bytes(&mut names)),
            (2, repeated_ints(&mut prefix_sizes)),
            (3, repeated_ints(&mut suffix_sizes)),
            (4, repeated_ints(&mut offsets)),
            (5, repeated_ints(&mut sizes)),
            (6, repeated_bytes(&mut endpoints)),
            (7, repeated_bytes(&mut rmrs)),
            (8, |x| prio = x),
        ];
        parse(proto, body, BodyKind::Request).await?;

        if offsets.len() != suffix_sizes.len() {
            return Err(Status::invalid_argument(format!(
                "bad offset count: wanted {}, got {}",
                suffix_sizes.len(),
                offsets.len()
            )));
        }
        if sizes.len() != suffix_sizes.len() {
            return Err(Status::invalid_argument(format!(
                "bad size count: wanted {}, got {}",
                suffix_sizes.len(),
                sizes.len()
            )));
        }
        if endpoints.len() > self.contexts.len() {
            return Err(Status::invalid_argument(format!(
                "{} RDMA endpoints > {} RDMA devices",
                endpoints.len(),
                self.contexts.len()
            )));
        }
        let regions: Vec<_> = decode_names(names, prefix_sizes, suffix_sizes)?
            .into_iter()
            .map(String::from_utf8)
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| Status::from_error(Box::new(e)))?
            .into_iter()
            .zip(offsets)
            .zip(sizes)
            .collect();
        let endpoints: Vec<_> = endpoints
            .into_iter()
            .enumerate()
            .filter_map(|(i, ep)| if ep.is_empty() { None } else { Some((i, ep)) })
            .collect();

        let mut extra_buf = get_bytes_mut();
        let mut queues = Vec::with_capacity(endpoints.len());
        for &(i, _) in &endpoints {
            let (ref ctx, ref pd) = self.contexts[i];
            let (ep, qp, cq) = make_qp(true, ctx, pd, MAX_QUEUE_DEPTH, prio)?;
            extra_buf.put_string(2, &ep);
            queues.push((qp, cq));
        }

        let mut mrz = Vec::with_capacity(regions.len());
        let mut slices = Vec::with_capacity(regions.len());
        let mut total_size = 0;
        let mut bodies = Vec::with_capacity(regions.len() + 1);
        let mut rdma_mask = Vec::with_capacity(regions.len());

        let max_entries = self.max_entries;
        let manifest = self.manifest();
        let no_mrs: MRs = Arc::new((0..self.contexts.len()).map(|_| None).collect());
        let mut peer_backed = false;
        let mut bytez = scopeguard::guard(self.bytez.lock().await, |bytez| {
            cleanup(bytez, max_entries);
        });
        for ((name, offset), size) in regions {
            let virtual_entry = manifest.as_ref().and_then(|m| m.resolve(&name));
            let (bytes, mrs) = if let Some(bytes) = virtual_entry {
                peer_backed = true;
                (bytes, no_mrs.clone())
            } else if let Some((bytes, mrs)) = bytez.get(&name) {
                (bytes.clone(), mrs.clone())
            } else {
                let (bytes, mrs) = match mmap_file(ROOT_DIR, &name, &self.contexts, &[]) {
                    Ok(v) => v,
                    Err(e) => {
                        if let Some(peer_serve) =
                            self.peer_serve.as_ref().filter(|_| manifest.is_none())
                        {
                            peer_serve.note_send_while_revoked();
                        }
                        return Err(e.into());
                    }
                };
                bytez.insert(name.clone(), (bytes.clone(), mrs.clone()));
                (bytes, mrs)
            };
            if offset + size > bytes.len() {
                return Err(Status::invalid_argument(format!(
                    "offset={offset} + size={size} exceeds {name} size={}",
                    bytes.len()
                )));
            }
            let use_rdma = endpoints.iter().any(|&(i, _)| mrs[i].is_some());
            if use_rdma && endpoints.iter().any(|&(i, _)| mrs[i].is_none()) {
                return Err(Status::invalid_argument(format!(
                    "{name} is not registered with a requested device"
                )));
            }
            if use_rdma && rmrs.is_empty() {
                for &(i, _) in &endpoints {
                    let rmr = mrs[i].as_ref().unwrap().remote();
                    let rmr = bincode::serialize(&rmr.slice(offset..offset + size))
                        .map_err(|e| Status::from_error(Box::new(e)))?;
                    extra_buf.put_string(3, &rmr);
                }
            }
            if use_rdma {
                let n = queues.len();
                let m = n * mrz.len();
                let step = size.div_ceil(n);
                let f = |(i, rmr)| {
                    get_rmr(rmr)
                        .is_ok_and(|rmr| rmr.len != 0 && rmr.len == cmp::min(step, size - i * step))
                };
                if m + n <= rmrs.len() && !rmrs.iter().skip(m).take(n).enumerate().all(f) {
                    return Err(Status::invalid_argument("bad remote memory region size"));
                }
                mrz.push((mrs, offset, size));
            } else {
                if size == 0 {
                    return Err(Status::invalid_argument("zero sizes are not supported"));
                }
                slices.push(bytes.slice(offset..offset + size));
                total_size += size;
            }
            rdma_mask.push(if use_rdma { 1 } else { 0 });
        }
        drop(bytez);
        extra_buf.put_string(4, &rdma_mask);

        if !rmrs.is_empty() && rmrs.len() != queues.len() * mrz.len() {
            return Err(Status::invalid_argument(format!(
                "bad remote memory region count: wanted {}, got {}",
                queues.len() * mrz.len(),
                rmrs.len()
            )));
        }
        if !mrz.is_empty() {
            tokio::spawn(async move {
                if let Err(e) = handle_rdma(endpoints, queues, mrz, rmrs).await {
                    log::error!("RDMA failed: {e}");
                }
            });
        }

        let frame_size = 0xFFFFFFFE - extra_buf.len();
        let mut pos = 0usize;
        let mut frame_body = |size| {
            let mut buf = get_bytes_mut();
            mem::swap(&mut buf, &mut extra_buf);
            buf.put_varints([0o12, size]);
            let len = buf.len() as u32 - 5 + size;
            buf[1..5].copy_from_slice(&len.to_be_bytes());
            Full::new(buf.freeze())
        };
        if slices.is_empty() {
            bodies.push(frame_body(0));
        }
        for bytes in slices {
            let size = bytes.len();
            let mut left = size;
            while left != 0 {
                let mut delta = pos.next_multiple_of(frame_size) - pos;
                if delta == 0 {
                    delta = frame_size;
                    bodies.push(frame_body(cmp::min(total_size - pos, delta) as u32));
                }
                delta = cmp::min(left, delta);
                const FULL_SIZE: usize = 0x7FFFFFFF;
                for a in (0..delta).step_by(FULL_SIZE) {
                    let b = cmp::min(a + FULL_SIZE, delta);
                    bodies.push(Full::new(bytes.slice(size - left + a..size - left + b)));
                }
                pos += delta;
                left -= delta;
            }
        }

        if peer_backed {
            let peer_serve = self.peer_serve.as_ref().unwrap();
            let permit = peer_serve
                .try_acquire_send_permit()
                .ok_or_else(|| Status::resource_exhausted("peer send concurrency limit reached"))?;
            return Ok(add_ok_trailer(PacedBody::new(
                concat_bodies(bodies),
                permit,
                peer_serve.pacer(),
            )));
        }
        Ok(add_ok_trailer(concat_bodies(bodies)))
    }

    pub async fn handle_register(self: Arc<Self>, body: body::Body) -> Result<body::Body, Status> {
        let mut path = Vec::new();
        let mut name = Vec::new();
        let mut device_indexes = Vec::new();
        let proto = proto![
            (1, bytes(&mut path)),
            (2, bytes(&mut name)),
            (3, bytes(&mut device_indexes)),
        ];
        parse(proto, body, BodyKind::Request).await?;

        let mut indexes = Vec::with_capacity(device_indexes.len());
        for i in device_indexes {
            if i as usize >= cmp::min(0x80, self.contexts.len()) {
                return Err(Status::invalid_argument(format!("bad device index {i}")));
            }
            indexes.push(i as usize);
        }
        let from_utf8 = |x| str::from_utf8(x).map_err(|e| Status::from_error(Box::new(e)));
        let (bytes, mrs) = mmap_file(
            from_utf8(&path)?,
            from_utf8(&name)?,
            &self.contexts,
            &indexes,
        )?;
        let max_entries = self.max_entries;
        scopeguard::guard(self.bytez.lock().await, |bytez| {
            cleanup(bytez, max_entries);
        })
        .insert(
            String::from_utf8(name).map_err(|e| Status::from_error(Box::new(e)))?,
            (bytes, mrs),
        );

        Ok(add_ok_trailer(freeze(get_bytes_mut())))
    }
}

async fn handle(
    sender: Arc<Sender>,
    request: http::Request<body::Body>,
) -> Result<http::Response<body::Body>, Infallible> {
    let path = request.uri().path();
    Ok(make_response(match path {
        LIST => sender.handle_list(request.into_body()).await,
        REGISTER => sender.handle_register(request.into_body()).await,
        SEND => sender.handle_send(request.into_body()).await,
        _ => Err(Status::unimplemented(format!("{path} not implemented"))),
    }))
}

#[derive(Clone)]
pub struct CopyService {
    sender: Arc<Sender>,
}

impl CopyService {
    pub fn new(sender: Sender) -> Self {
        Self {
            sender: Arc::new(sender),
        }
    }

    pub fn from_arc(sender: Arc<Sender>) -> Self {
        Self { sender }
    }
}

impl NamedService for CopyService {
    const NAME: &'static str = "copy.Copy";
}

impl Service<http::Request<body::Body>> for CopyService {
    type Response = http::Response<body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Ok(()).into()
    }

    fn call(&mut self, request: http::Request<body::Body>) -> Self::Future {
        Box::pin(handle(self.sender.clone(), request))
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use std::{
        os::unix::fs::symlink,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "xai-copy-path-test-{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn checkpoint_file_open_stays_within_root() {
        let assert_denied = |result: io::Result<fs::File>| {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::PermissionDenied);
        };
        let temp = TestDir::new();
        let root = temp.0.join("checkpoints");
        let nested = root.join("run");
        let staging = root.join(".staging");
        let outside = temp.0.join("outside");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir(&staging).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(nested.join("weights.bin"), b"weights").unwrap();
        fs::write(staging.join("weights.bin"), b"staging").unwrap();
        fs::write(outside.join("secret"), b"secret").unwrap();

        let send_prefix = PathBuf::from(format!("{}/", root.display()));
        let file = open_file_within(&root, &send_prefix, Path::new("run/weights.bin")).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 7);

        let register_prefix = PathBuf::from(format!("{}/.", root.display()));
        let file =
            open_file_within(&root, &register_prefix, Path::new("staging/weights.bin")).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 7);

        assert_denied(open_file_within(
            &root,
            &send_prefix,
            Path::new("../outside/secret"),
        ));
        assert_denied(open_file_within(
            &root,
            Path::new(""),
            &outside.join("secret"),
        ));
        assert_denied(open_file_within(&root, &outside, Path::new("/secret")));

        symlink(outside.join("secret"), root.join("linked-secret")).unwrap();
        assert_denied(open_file_within(
            &root,
            &send_prefix,
            Path::new("linked-secret"),
        ));

        let outside_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(outside.join("secret"))
            .unwrap();
        assert_eq!(
            ensure_open_file_within(&root.canonicalize().unwrap(), &outside_file)
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
    }
}

#[cfg(all(test, feature = "rdma-tests"))]
mod tests {
    use super::*;
    use crate::proto_parser::repeated_ints;
    use std::path::PathBuf;

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(tag: &str, data: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "recsys_p2p_{tag}_{}_{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            fs::write(&path, data).unwrap();
            Self { path }
        }

        fn path_str(&self) -> String {
            self.path.to_str().unwrap().to_string()
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn backing_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn peer_sender(max_sends: usize, rate: Option<f64>) -> Arc<Sender> {
        Arc::new(Sender::new(DEFAULT_MAX_ENTRIES).with_peer_serve(max_sends, rate))
    }

    fn publish(sender: &Sender, file: &TempFile) {
        sender
            .publish_manifest(
                vec![
                    ManifestSpec {
                        name: "elapsed_test/run/emb_table/c/0/0".into(),
                        file: file.path_str(),
                        offset: 0,
                        size: 512,
                    },
                    ManifestSpec {
                        name: "elapsed_test/run/emb_table/c/1/0".into(),
                        file: file.path_str(),
                        offset: 512,
                        size: 512,
                    },
                ],
                vec![("elapsed_test/run/checksums.0.json".into(), b"{}".to_vec())],
            )
            .unwrap();
    }

    fn empty_frame_body() -> body::Body {
        body::Body::new(freeze(get_bytes_mut()))
    }

    fn send_request_body(names: Vec<&str>, offsets: Vec<usize>, sizes: Vec<usize>) -> body::Body {
        let (all_names, prefix_sizes, suffix_sizes) =
            encode_names(names.into_iter().map(|s| s.as_bytes().to_vec()).collect());
        let mut buf = get_bytes_mut();
        buf.put_string(1, &all_names);
        buf.put_repeated_ints(
            [2, 3, 4, 5],
            [&prefix_sizes, &suffix_sizes, &offsets, &sizes],
        );
        body::Body::new(freeze(buf))
    }

    async fn list_names(sender: &Arc<Sender>) -> Vec<(String, usize)> {
        let body = sender
            .clone()
            .handle_list(empty_frame_body())
            .await
            .unwrap();
        let mut all_names = Vec::new();
        let mut prefix_sizes = Vec::<usize>::new();
        let mut suffix_sizes = Vec::<usize>::new();
        let mut sizes = Vec::<usize>::new();
        parse(
            proto![
                (1, bytes(&mut all_names)),
                (2, repeated_ints(&mut prefix_sizes)),
                (3, repeated_ints(&mut suffix_sizes)),
                (4, repeated_ints(&mut sizes)),
            ],
            body,
            BodyKind::Response,
        )
        .await
        .unwrap();
        decode_names(all_names, prefix_sizes, suffix_sizes)
            .unwrap()
            .into_iter()
            .map(|n| String::from_utf8(n).unwrap())
            .zip(sizes)
            .collect()
    }

    async fn send_and_read(
        sender: &Arc<Sender>,
        names: Vec<&str>,
        offsets: Vec<usize>,
        sizes: Vec<usize>,
    ) -> Result<Vec<u8>, Status> {
        let body = sender
            .clone()
            .handle_send(send_request_body(names, offsets, sizes))
            .await?;
        let mut data = Vec::new();
        let mut rdma_mask = Vec::new();
        parse(
            proto![(1, bytes(&mut data)), (4, bytes(&mut rdma_mask))],
            body,
            BodyKind::Response,
        )
        .await?;
        assert!(
            rdma_mask.iter().all(|&m| m == 0),
            "virtual entries must not use RDMA"
        );
        Ok(data)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_includes_manifest_entries() {
        let file = TempFile::new("list", &backing_data(1024));
        let sender = peer_sender(4, None);
        publish(&sender, &file);

        let names = list_names(&sender).await;
        for (name, size) in [
            ("elapsed_test/run/emb_table/c/0/0", 512),
            ("elapsed_test/run/emb_table/c/1/0", 512),
            ("elapsed_test/run/checksums.0.json", 2),
        ] {
            assert!(
                names.contains(&(name.to_string(), size)),
                "missing {name} in {names:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn send_serves_virtual_slices_and_blobs() {
        let data = backing_data(1024);
        let file = TempFile::new("send", &data);
        let sender = peer_sender(4, None);
        publish(&sender, &file);

        let got = send_and_read(
            &sender,
            vec!["elapsed_test/run/emb_table/c/1/0"],
            vec![10],
            vec![100],
        )
        .await
        .unwrap();
        assert_eq!(got, &data[512 + 10..512 + 110]);

        let got = send_and_read(
            &sender,
            vec!["elapsed_test/run/checksums.0.json"],
            vec![0],
            vec![2],
        )
        .await
        .unwrap();
        assert_eq!(got, b"{}");

        let err = send_and_read(
            &sender,
            vec!["elapsed_test/run/emb_table/c/1/0"],
            vec![500],
            vec![100],
        )
        .await
        .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn revoke_unpublishes_and_send_fails() {
        let file = TempFile::new("revoke", &backing_data(1024));
        let sender = peer_sender(4, None);
        publish(&sender, &file);
        assert!(sender.revoke_manifest(Duration::from_secs(1)));

        let names = list_names(&sender).await;
        assert!(names.iter().all(|(n, _)| !n.starts_with("elapsed_test/")));
        assert!(
            send_and_read(
                &sender,
                vec!["elapsed_test/run/emb_table/c/0/0"],
                vec![0],
                vec![1]
            )
            .await
            .is_err()
        );
        assert!(sender.revoke_manifest(Duration::ZERO));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn revoke_waits_for_inflight_sends() {
        let file = TempFile::new("drain", &backing_data(1024));
        let sender = peer_sender(4, None);
        publish(&sender, &file);

        let body = sender
            .clone()
            .handle_send(send_request_body(
                vec!["elapsed_test/run/emb_table/c/0/0"],
                vec![0],
                vec![512],
            ))
            .await
            .unwrap();

        let s = sender.clone();
        let drained = task::spawn_blocking(move || s.revoke_manifest(Duration::from_millis(100)))
            .await
            .unwrap();
        assert!(!drained, "in-flight send must block the drain");

        drop(body);
        let s = sender.clone();
        publish(&sender, &file);
        let drained = task::spawn_blocking(move || s.revoke_manifest(Duration::from_secs(5)))
            .await
            .unwrap();
        assert!(drained);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn send_concurrency_limit() {
        let file = TempFile::new("limit", &backing_data(1024));
        let sender = peer_sender(1, None);
        publish(&sender, &file);

        let names = vec!["elapsed_test/run/emb_table/c/0/0"];
        let held = sender
            .clone()
            .handle_send(send_request_body(names.clone(), vec![0], vec![512]))
            .await
            .unwrap();
        let err = sender
            .clone()
            .handle_send(send_request_body(names.clone(), vec![0], vec![512]))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::ResourceExhausted);

        drop(held);
        assert!(
            sender
                .clone()
                .handle_send(send_request_body(names, vec![0], vec![512]))
                .await
                .is_ok()
        );
    }

    #[test]
    fn publish_refused_and_revoke_noop_when_disabled() {
        let disabled = Sender::new(DEFAULT_MAX_ENTRIES);
        assert!(disabled.publish_manifest(vec![], vec![]).is_err());
        assert!(disabled.revoke_manifest(Duration::ZERO));
    }
}
