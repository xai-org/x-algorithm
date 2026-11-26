// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame};
use lazy_static::lazy_static;
use prometheus::{
    IntCounter, IntCounterVec, IntGauge, register_int_counter, register_int_counter_vec,
    register_int_gauge,
};
use std::{
    cmp,
    collections::BTreeMap,
    convert::Infallible,
    fs,
    future::Future,
    io,
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

lazy_static! {
    pub static ref PEER_SERVE_BYTES_TOTAL: IntCounter = register_int_counter!(
        "recsys_engine_peer_serve_bytes_total",
        "Checkpoint bytes served to peer pods via manifest-backed /copy.Copy/Send"
    )
    .unwrap();
    pub static ref PEER_SERVE_SENDS_INFLIGHT: IntGauge = register_int_gauge!(
        "recsys_engine_peer_serve_sends_inflight",
        "Manifest-backed peer sends currently streaming"
    )
    .unwrap();
    pub static ref PEER_SERVE_REJECTED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "recsys_engine_peer_serve_rejected_total",
        "Peer sends rejected before streaming (limit = concurrency permit \
         exhausted; revoked = request while no manifest is published)",
        &["reason"]
    )
    .unwrap();
}

#[derive(Clone)]
pub(crate) struct PeerServeMetrics {
    bytes_total: IntCounter,
    sends_inflight: IntGauge,
    rejected_total: IntCounterVec,
}

impl PeerServeMetrics {
    fn global() -> Self {
        Self {
            bytes_total: PEER_SERVE_BYTES_TOTAL.clone(),
            sends_inflight: PEER_SERVE_SENDS_INFLIGHT.clone(),
            rejected_total: PEER_SERVE_REJECTED_TOTAL.clone(),
        }
    }

    #[cfg(test)]
    fn isolated() -> Self {
        Self {
            bytes_total: IntCounter::new(
                "recsys_engine_peer_serve_bytes_total_isolated",
                "test-local peer serve bytes",
            )
            .unwrap(),
            sends_inflight: IntGauge::new(
                "recsys_engine_peer_serve_sends_inflight_isolated",
                "test-local peer serve inflight",
            )
            .unwrap(),
            rejected_total: IntCounterVec::new(
                prometheus::Opts::new(
                    "recsys_engine_peer_serve_rejected_total_isolated",
                    "test-local peer serve rejections",
                ),
                &["reason"],
            )
            .unwrap(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ManifestSpec {
    pub name: String,
    pub file: String,
    pub offset: usize,
    pub size: usize,
}

struct CachedMmap {
    mmap: Option<memmap2::Mmap>,
}

impl AsRef<[u8]> for CachedMmap {
    fn as_ref(&self) -> &[u8] {
        self.mmap
            .as_ref()
            .expect("CachedMmap used after drop")
            .as_ref()
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct FileIdentity {
    ino: u64,
    len: u64,
}

#[cfg(unix)]
fn file_identity(file: &fs::File) -> io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    let md = file.metadata()?;
    Ok(FileIdentity {
        ino: md.ino(),
        len: md.len(),
    })
}

#[cfg(not(unix))]
fn file_identity(file: &fs::File) -> io::Result<FileIdentity> {
    let md = file.metadata()?;
    Ok(FileIdentity {
        ino: 0,
        len: md.len(),
    })
}

static MMAP_CACHE: StdMutex<BTreeMap<String, (FileIdentity, Bytes)>> =
    StdMutex::new(BTreeMap::new());

fn cached_file_bytes(path: &str) -> io::Result<Bytes> {
    let file = fs::File::open(path)?;
    let ident = file_identity(&file)?;
    let mut cache = MMAP_CACHE.lock().unwrap();
    if let Some((cached_ident, bytes)) = cache.get(path) {
        if *cached_ident == ident {
            return Ok(bytes.clone());
        }
        log::info!("manifest mmap cache: {path} changed (inode/len); remapping");
    }
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let bytes = Bytes::from_owner(CachedMmap { mmap: Some(mmap) });
    cache.insert(path.to_string(), (ident, bytes.clone()));
    Ok(bytes)
}

struct LeasedSlice {
    bytes: Bytes,
    _lease: Arc<()>,
}

impl AsRef<[u8]> for LeasedSlice {
    fn as_ref(&self) -> &[u8] {
        self.bytes.as_ref()
    }
}

const UNMAP_SLICE_BYTES: usize = 4 << 30;

#[cfg(unix)]
fn unmap_in_slices(mmap: memmap2::Mmap, slice_bytes: usize) {
    let ptr = mmap.as_ptr() as usize;
    let len = mmap.len();
    std::mem::forget(mmap);
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    let slice = cmp::max(slice_bytes.next_multiple_of(page), page);
    let mut off = 0;
    while off < len {
        let n = cmp::min(slice, len - off);
        unsafe {
            libc::munmap((ptr + off) as *mut libc::c_void, n);
        }
        off += n;
        std::thread::yield_now();
    }
}

#[cfg(unix)]
static UNMAP_REAPER: std::sync::LazyLock<Option<std::sync::mpsc::Sender<memmap2::Mmap>>> =
    std::sync::LazyLock::new(|| {
        let (tx, rx) = std::sync::mpsc::channel::<memmap2::Mmap>();
        std::thread::Builder::new()
            .name("manifest-unmap".into())
            .spawn(move || {
                for mmap in rx {
                    unmap_in_slices(mmap, UNMAP_SLICE_BYTES);
                }
            })
            .inspect_err(|e| log::warn!("manifest-unmap reaper spawn failed: {e}"))
            .ok()
            .map(|_handle| tx)
    });

impl Drop for CachedMmap {
    fn drop(&mut self) {
        let Some(mmap) = self.mmap.take() else {
            return;
        };
        #[cfg(unix)]
        {
            let mmap = match UNMAP_REAPER.as_ref() {
                Some(tx) => match tx.send(mmap) {
                    Ok(()) => return,
                    Err(std::sync::mpsc::SendError(mmap)) => mmap,
                },
                None => mmap,
            };
            log::warn!("manifest-unmap reaper unavailable; sliced unmap on calling thread");
            unmap_in_slices(mmap, UNMAP_SLICE_BYTES);
        }
        #[cfg(not(unix))]
        drop(mmap);
    }
}

pub struct PeerManifest {
    entries: BTreeMap<String, (Bytes, usize, usize)>,
    blobs: BTreeMap<String, Bytes>,
    lease: Arc<()>,
}

impl PeerManifest {
    pub fn resolve(&self, name: &str) -> Option<Bytes> {
        self.entries
            .get(name)
            .map(|(bytes, base, size)| bytes.slice(*base..*base + *size))
            .or_else(|| self.blobs.get(name).cloned())
    }

    pub fn list(&self) -> impl Iterator<Item = (Vec<u8>, usize)> + '_ {
        self.entries
            .iter()
            .map(|(name, (_, _, size))| (name.clone().into_bytes(), *size))
            .chain(
                self.blobs
                    .iter()
                    .map(|(name, blob)| (name.clone().into_bytes(), blob.len())),
            )
    }
}

pub struct PeerPacer {
    bytes_per_sec: f64,
    state: StdMutex<(Instant, f64)>,
}

impl PeerPacer {
    const BURST_SECS: f64 = 0.25;

    pub fn new(bytes_per_sec: f64) -> Self {
        Self {
            bytes_per_sec,
            state: StdMutex::new((Instant::now(), bytes_per_sec * Self::BURST_SECS)),
        }
    }

    fn consume(&self, n: usize) -> Duration {
        let mut state = self.state.lock().unwrap();
        let (last, tokens) = *state;
        let now = Instant::now();
        let tokens = (tokens + now.duration_since(last).as_secs_f64() * self.bytes_per_sec)
            .min(self.bytes_per_sec * Self::BURST_SECS);
        let tokens = tokens - n as f64;
        *state = (now, tokens);
        if tokens >= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(-tokens / self.bytes_per_sec)
        }
    }
}

pub struct PeerServeState {
    manifest: StdMutex<Option<Arc<PeerManifest>>>,
    permits: Arc<Semaphore>,
    pacer: Option<Arc<PeerPacer>>,
    metrics: PeerServeMetrics,
}

impl PeerServeState {
    pub fn new(max_concurrent_sends: usize, rate_limit_bytes_per_sec: Option<f64>) -> Self {
        Self::new_with_metrics(
            max_concurrent_sends,
            rate_limit_bytes_per_sec,
            PeerServeMetrics::global(),
        )
    }

    fn new_with_metrics(
        max_concurrent_sends: usize,
        rate_limit_bytes_per_sec: Option<f64>,
        metrics: PeerServeMetrics,
    ) -> Self {
        Self {
            manifest: StdMutex::new(None),
            permits: Arc::new(Semaphore::new(max_concurrent_sends.max(1))),
            pacer: rate_limit_bytes_per_sec
                .filter(|r| *r > 0.0)
                .map(|r| Arc::new(PeerPacer::new(r))),
            metrics,
        }
    }

    pub fn publish(
        &self,
        entries: Vec<ManifestSpec>,
        blobs: Vec<(String, Vec<u8>)>,
    ) -> io::Result<()> {
        let blobs = blobs
            .into_iter()
            .map(|(name, data)| (name, Bytes::from(data)))
            .collect();
        self.publish_with_memory(entries, blobs)
    }

    pub fn publish_with_memory(
        &self,
        entries: Vec<ManifestSpec>,
        memory: Vec<(String, Bytes)>,
    ) -> io::Result<()> {
        let lease = Arc::new(());
        let mut files: BTreeMap<String, Bytes> = BTreeMap::new();
        let mut manifest_entries = BTreeMap::new();
        for spec in entries {
            if !files.contains_key(&spec.file) {
                let backing = cached_file_bytes(&spec.file)?;
                files.insert(
                    spec.file.clone(),
                    Bytes::from_owner(LeasedSlice {
                        bytes: backing,
                        _lease: lease.clone(),
                    }),
                );
            }
            let bytes = &files[&spec.file];
            if spec.offset + spec.size > bytes.len() {
                return Err(io::Error::other(format!(
                    "manifest entry {}: offset={} + size={} exceeds {} len={}",
                    spec.name,
                    spec.offset,
                    spec.size,
                    spec.file,
                    bytes.len()
                )));
            }
            manifest_entries.insert(spec.name, (bytes.clone(), spec.offset, spec.size));
        }
        let blobs = memory
            .into_iter()
            .map(|(name, bytes)| {
                let leased = Bytes::from_owner(LeasedSlice {
                    bytes,
                    _lease: lease.clone(),
                });
                (name, leased)
            })
            .collect();
        *self.manifest.lock().unwrap() = Some(Arc::new(PeerManifest {
            entries: manifest_entries,
            blobs,
            lease,
        }));
        Ok(())
    }

    pub fn revoke(&self, drain_timeout: Duration) -> bool {
        let Some(manifest) = self.manifest.lock().unwrap().take() else {
            return true;
        };
        let lease = manifest.lease.clone();
        drop(manifest);
        let deadline = Instant::now() + drain_timeout;
        while Arc::strong_count(&lease) > 1 {
            if Instant::now() >= deadline {
                log::warn!(
                    "peer manifest revoke: {} lease(s) still held after {:?}; \
                     in-flight peer reads may see torn data (peers verify checksums)",
                    Arc::strong_count(&lease) - 1,
                    drain_timeout
                );
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        true
    }

    pub fn unpublish(&self) {
        self.manifest.lock().unwrap().take();
    }

    pub fn manifest(&self) -> Option<Arc<PeerManifest>> {
        self.manifest.lock().unwrap().clone()
    }

    pub fn try_acquire_send_permit(&self) -> Option<OwnedSemaphorePermit> {
        let permit = self.permits.clone().try_acquire_owned().ok();
        if permit.is_none() {
            self.metrics
                .rejected_total
                .with_label_values(&["limit"])
                .inc();
        }
        permit
    }

    pub fn note_send_while_revoked(&self) {
        self.metrics
            .rejected_total
            .with_label_values(&["revoked"])
            .inc();
    }

    pub fn pacer(&self) -> Option<Arc<PeerPacer>> {
        self.pacer.clone()
    }
}

pub const PEER_PACE_CHUNK: usize = 8 << 20;

pub struct PacedBody {
    inner: Pin<Box<dyn HttpBody<Data = Bytes, Error = Infallible> + Send>>,
    pending: Bytes,
    sleep: Option<Pin<Box<tokio::time::Sleep>>>,
    pacer: Option<Arc<PeerPacer>>,
    metrics: PeerServeMetrics,
    _permit: OwnedSemaphorePermit,
}

impl PacedBody {
    pub fn new<B: HttpBody<Data = Bytes, Error = Infallible> + Send + 'static>(
        inner: B,
        permit: OwnedSemaphorePermit,
        pacer: Option<Arc<PeerPacer>>,
    ) -> Self {
        Self::with_metrics(inner, permit, pacer, PeerServeMetrics::global())
    }

    fn with_metrics<B: HttpBody<Data = Bytes, Error = Infallible> + Send + 'static>(
        inner: B,
        permit: OwnedSemaphorePermit,
        pacer: Option<Arc<PeerPacer>>,
        metrics: PeerServeMetrics,
    ) -> Self {
        metrics.sends_inflight.inc();
        Self {
            inner: Box::pin(inner),
            pending: Bytes::new(),
            sleep: None,
            pacer,
            metrics,
            _permit: permit,
        }
    }

    fn pace(&mut self, n: usize) {
        if let Some(pacer) = &self.pacer {
            let wait = pacer.consume(n);
            if !wait.is_zero() {
                self.sleep = Some(Box::pin(tokio::time::sleep(wait)));
            }
        }
    }
}

impl Drop for PacedBody {
    fn drop(&mut self) {
        self.metrics.sends_inflight.dec();
    }
}

impl HttpBody for PacedBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        loop {
            if let Some(sleep) = &mut self.sleep {
                match sleep.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(()) => self.sleep = None,
                }
            }
            if !self.pending.is_empty() {
                let n = cmp::min(self.pending.len(), PEER_PACE_CHUNK);
                let chunk = self.pending.split_to(n);
                self.pace(n);
                self.metrics.bytes_total.inc_by(n as u64);
                return Poll::Ready(Some(Ok(Frame::data(chunk))));
            }
            match self.inner.as_mut().poll_frame(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                    Ok(data) => self.pending = data,
                    Err(frame) => return Poll::Ready(Some(Ok(frame))),
                },
                Poll::Ready(Some(Err(e))) => match e {},
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::Full;
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

    #[cfg(unix)]
    #[test]
    fn unmap_in_slices_covers_whole_mapping() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
        for slice_pages in [2usize, 3, 100] {
            let f = TempFile::new("unmap", &vec![7u8; 8 * page]);
            let file = fs::File::open(&f.path).unwrap();
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
            assert_eq!(mmap.len(), 8 * page);
            unmap_in_slices(mmap, slice_pages * page);
        }
    }

    #[test]
    fn mmap_cache_reuses_and_replaces_mappings() {
        let f = TempFile::new("cache", &vec![1u8; 1 << 20]);
        let a = cached_file_bytes(&f.path_str()).unwrap();
        let b = cached_file_bytes(&f.path_str()).unwrap();
        assert_eq!(a.as_ptr(), b.as_ptr(), "unchanged file must reuse mapping");

        fs::remove_file(&f.path).unwrap();
        fs::write(&f.path, vec![2u8; 2 << 20]).unwrap();
        let c = cached_file_bytes(&f.path_str()).unwrap();
        assert_ne!(a.as_ptr(), c.as_ptr(), "recreated file must remap");
        assert_eq!(c.len(), 2 << 20);
        assert_eq!(a.len(), 1 << 20);
        assert_eq!(a[0], 1u8);
        assert_eq!(c[0], 2u8);
    }

    #[test]
    fn leased_slice_drop_releases_lease_without_unmap() {
        let f = TempFile::new("lease", &vec![3u8; 1 << 20]);
        let backing = cached_file_bytes(&f.path_str()).unwrap();
        let ptr = backing.as_ptr();
        let lease = Arc::new(());
        let wrapped = Bytes::from_owner(LeasedSlice {
            bytes: backing,
            _lease: lease.clone(),
        });
        let slice = wrapped.slice(16..32);
        drop(wrapped);
        assert_eq!(Arc::strong_count(&lease), 2, "slice still holds lease");
        let start = Instant::now();
        drop(slice);
        assert!(start.elapsed() < Duration::from_millis(100));
        assert_eq!(Arc::strong_count(&lease), 1, "lease released on drop");
        let again = cached_file_bytes(&f.path_str()).unwrap();
        assert_eq!(again.as_ptr(), ptr);
    }

    fn backing_data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn publish(state: &PeerServeState, file: &TempFile) {
        state
            .publish(
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

    #[test]
    fn list_names_and_sizes() {
        let file = TempFile::new("list", &backing_data(1024));
        let state = PeerServeState::new(4, None);
        publish(&state, &file);

        let names: Vec<(String, usize)> = state
            .manifest()
            .unwrap()
            .list()
            .map(|(n, s)| (String::from_utf8(n).unwrap(), s))
            .collect();
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

    #[test]
    fn resolve_slices_blobs_and_unknown_names() {
        let data = backing_data(1024);
        let file = TempFile::new("resolve", &data);
        let state = PeerServeState::new(4, None);
        publish(&state, &file);
        let manifest = state.manifest().unwrap();

        let piece = manifest
            .resolve("elapsed_test/run/emb_table/c/1/0")
            .unwrap();
        assert_eq!(&piece[..], &data[512..1024]);
        let blob = manifest
            .resolve("elapsed_test/run/checksums.0.json")
            .unwrap();
        assert_eq!(&blob[..], b"{}");
        assert!(
            manifest
                .resolve("elapsed_test/run/emb_table/c/2/0")
                .is_none()
        );
        assert!(manifest.resolve("../../etc/passwd").is_none());
        assert!(manifest.resolve("elapsed_test/run/emb_table").is_none());
    }

    #[test]
    fn publish_validates_bounds() {
        let file = TempFile::new("bounds", &backing_data(100));
        let state = PeerServeState::new(1, None);
        let err = state
            .publish(
                vec![ManifestSpec {
                    name: "x/c/0/0".into(),
                    file: file.path_str(),
                    offset: 50,
                    size: 100,
                }],
                vec![],
            )
            .unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
        assert!(state.manifest().is_none());
    }

    #[test]
    fn revoke_unpublishes_and_is_idempotent() {
        let file = TempFile::new("revoke", &backing_data(1024));
        let state = PeerServeState::new(4, None);
        publish(&state, &file);
        assert!(state.revoke(Duration::from_secs(1)));
        assert!(state.manifest().is_none());
        assert!(state.revoke(Duration::ZERO));
    }

    #[test]
    fn revoke_waits_for_inflight_reads() {
        let file = TempFile::new("drain", &backing_data(1024));
        let state = PeerServeState::new(4, None);
        publish(&state, &file);

        let held = state
            .manifest()
            .unwrap()
            .resolve("elapsed_test/run/emb_table/c/0/0")
            .unwrap();
        assert!(
            !state.revoke(Duration::from_millis(100)),
            "in-flight read must block the drain"
        );

        drop(held);
        publish(&state, &file);
        assert!(state.revoke(Duration::from_secs(5)));
    }

    struct Owned(Arc<Vec<u8>>);

    impl AsRef<[u8]> for Owned {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    #[test]
    fn memory_entries_list_resolve_and_release_their_owner() {
        let buf = Arc::new(backing_data(1024));
        let state = PeerServeState::new(4, None);
        state
            .publish_with_memory(
                vec![],
                vec![
                    (
                        "elapsed_test/run/w/c/0".into(),
                        Bytes::from_owner(Owned(buf.clone())),
                    ),
                    (
                        "elapsed_test/run/checksums.0.json".into(),
                        Bytes::from_static(b"{}"),
                    ),
                ],
            )
            .unwrap();
        let manifest = state.manifest().unwrap();
        assert_eq!(
            vec![
                ("elapsed_test/run/checksums.0.json".to_owned(), 2),
                ("elapsed_test/run/w/c/0".to_owned(), 1024),
            ],
            manifest
                .list()
                .map(|(n, s)| (String::from_utf8(n).unwrap(), s))
                .collect::<Vec<_>>()
        );

        let held = manifest.resolve("elapsed_test/run/w/c/0").unwrap();
        assert_eq!(&buf[..], &held[..]);
        drop(manifest);
        state.unpublish();
        assert!(state.manifest().is_none());
        assert_eq!(
            2,
            Arc::strong_count(&buf),
            "the served slice keeps its owner"
        );
        drop(held);
        assert_eq!(
            1,
            Arc::strong_count(&buf),
            "owner released with the last slice"
        );
    }

    #[test]
    fn send_permits_bound_concurrency() {
        let state = PeerServeState::new(1, None);
        let held = state.try_acquire_send_permit().unwrap();
        assert!(state.try_acquire_send_permit().is_none());
        drop(held);
        assert!(state.try_acquire_send_permit().is_some());
    }

    #[test]
    fn pacer_math() {
        let pacer = PeerPacer::new(1_000_000.0);
        assert_eq!(pacer.consume(250_000), Duration::ZERO);
        let wait = pacer.consume(1_000_000);
        assert!(
            (0.5..=1.05).contains(&wait.as_secs_f64()),
            "expected ~1s deficit wait, got {wait:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn paced_body_rechunks_and_preserves_bytes() {
        use http_body_util::BodyExt;
        let data: Bytes = backing_data(PEER_PACE_CHUNK * 2 + 1234).into();
        let metrics = PeerServeMetrics::isolated();
        let permit = Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap();
        let mut body =
            PacedBody::with_metrics(Full::new(data.clone()), permit, None, metrics.clone());
        assert_eq!(metrics.sends_inflight.get(), 1);

        let mut chunks = Vec::new();
        while let Some(frame) = body.frame().await {
            if let Ok(chunk) = frame.unwrap().into_data() {
                chunks.push(chunk);
            }
        }
        assert_eq!(
            chunks.iter().map(Bytes::len).collect::<Vec<_>>(),
            vec![PEER_PACE_CHUNK, PEER_PACE_CHUNK, 1234]
        );
        assert_eq!(chunks.concat(), data);

        assert_eq!(metrics.bytes_total.get(), data.len() as u64);
        drop(body);
        assert_eq!(metrics.sends_inflight.get(), 0);
    }

    #[test]
    fn rejected_permit_counted_as_limit() {
        let metrics = PeerServeMetrics::isolated();
        let state = PeerServeState::new_with_metrics(1, None, metrics.clone());
        let held = state.try_acquire_send_permit().unwrap();
        assert!(state.try_acquire_send_permit().is_none());
        assert_eq!(
            metrics.rejected_total.with_label_values(&["limit"]).get(),
            1
        );
        drop(held);

        state.note_send_while_revoked();
        assert_eq!(
            metrics.rejected_total.with_label_values(&["revoked"]).get(),
            1
        );
    }
}
