// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::{
    checkpoint_policy::{
        Decide, Inventory, Ledger, Mode, NfsPolicy, Offer, Peer, Policy, SEQ_WIDTH, Source,
        binding_key, candidate_offer, is_real_prefix, parse_prefix, prefix_of, same_run,
    },
    r#const::{LIST, REGISTER, SEND},
    grpc_util::{add_ok_trailer, freeze, get_bytes_mut, make_response},
    proto,
    proto_parser::{
        BodyKind, BytesMutProtoExt, bytes, decode_names, encode_names, parse, proto,
        repeated_bytes, repeated_ints,
    },
};
use futures::future::BoxFuture;
use log::{info, warn};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    fs, io,
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tonic::{
    Status, body,
    codegen::{Service, http},
    server::NamedService,
    transport::server::{TcpConnectInfo, TlsConnectInfo},
};

const GATEWAY_DIR: &str = "gateway";
const HELD_DIR: &str = "held";
const STATE_FILE: &str = "state.json";

#[derive(clap::Args, Debug, Clone, Default)]
pub struct GatewayArgs {
    #[clap(long)]
    pub policy_url: Option<String>,

    #[clap(long, requires = "policy_url", default_value_t = 120)]
    pub policy_fresh_secs: u64,

    #[clap(long, requires = "policy_url", default_value_t = 10)]
    pub policy_poll_secs: u64,

    #[clap(long, requires = "policy_url")]
    pub policy_tls_ca: Option<String>,
    #[clap(long, requires_all = ["policy_url", "policy_tls_key", "policy_tls_ca"])]
    pub policy_tls_cert: Option<String>,
    #[clap(long, requires_all = ["policy_url", "policy_tls_cert", "policy_tls_ca"])]
    pub policy_tls_key: Option<String>,
    #[clap(long, requires = "policy_url")]
    pub policy_tls_server_name: Option<String>,

    #[clap(long, requires = "policy_url")]
    pub gateway_state_dir: Option<String>,

    #[clap(long, requires = "policy_url", default_value_t = 15)]
    pub pod_refresh_secs: u64,

    #[clap(long, requires = "policy_url", default_value_t = POD_STALE_SECS)]
    pub pod_stale_secs: u64,

    #[clap(long, requires = "policy_url", default_value_t = UNKNOWN_PEER_SECS)]
    pub unknown_peer_secs: u64,
}

const POD_STALE_SECS: u64 = 60;
const UNKNOWN_PEER_SECS: u64 = 300;

impl GatewayArgs {
    pub fn enabled(&self) -> bool {
        self.policy_url.is_some()
    }

    pub fn tls(&self) -> Result<Option<crate::tls::ClientTlsOptions>, String> {
        let Some(url) = &self.policy_url else {
            return Ok(None);
        };
        let any_tls = self.policy_tls_ca.is_some()
            || self.policy_tls_cert.is_some()
            || self.policy_tls_key.is_some()
            || self.policy_tls_server_name.is_some();
        if !url.starts_with("https://") {
            return match any_tls {
                true => Err(format!(
                    "--policy-tls-* apply to an https:// --policy-url; {url} is plaintext"
                )),
                false => Ok(None),
            };
        }
        let ca_path = self
            .policy_tls_ca
            .clone()
            .ok_or_else(|| format!("the https --policy-url {url} needs --policy-tls-ca"))?;
        Ok(Some(crate::tls::ClientTlsOptions {
            ca_path,
            cert_path: self.policy_tls_cert.clone(),
            key_path: self.policy_tls_key.clone(),
            server_name: self.policy_tls_server_name.clone(),
        }))
    }

    pub fn binding(&self) -> Option<String> {
        let url = self.policy_url.as_deref()?;
        let (_, path) = split_url(url).ok()?;
        let mut parts = path
            .split('/')
            .skip_while(|p| *p != "prod-trainers")
            .skip(1);
        parts.next().filter(|id| !id.is_empty()).map(str::to_string)
    }
}

pub trait PeerDirectory: Send + Sync {
    fn lookup(&self, ip: IpAddr) -> Option<(i64, String)>;
}

pub struct NoPeers;

impl PeerDirectory for NoPeers {
    fn lookup(&self, _: IpAddr) -> Option<(i64, String)> {
        None
    }
}

#[derive(Default)]
pub struct StaticPeers(pub HashMap<IpAddr, (i64, String)>);

impl PeerDirectory for StaticPeers {
    fn lookup(&self, ip: IpAddr) -> Option<(i64, String)> {
        self.0.get(&ip).cloned()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PeerAddr(pub SocketAddr);

fn peer_ip<B>(request: &http::Request<B>) -> Option<IpAddr> {
    let ext = request.extensions();
    ext.get::<PeerAddr>()
        .map(|p| p.0)
        .or_else(|| {
            ext.get::<TcpConnectInfo>()
                .and_then(TcpConnectInfo::remote_addr)
        })
        .or_else(|| {
            ext.get::<TlsConnectInfo<TcpConnectInfo>>()
                .and_then(|i| i.get_ref().remote_addr())
        })
        .map(|a| a.ip())
}

#[derive(Clone, Debug)]
pub struct GatewayConfig {
    pub root: PathBuf,
    pub state_dir: Option<PathBuf>,
    pub server_index: usize,
    pub policy_fresh: Duration,
    pub name: String,
    pub binding: Option<String>,
    pub unknown_peer_ttl: Duration,
}

#[derive(Default)]
struct State {
    policy: Option<Policy>,
    read_at: Option<Instant>,
    ledger: Ledger,
    latest: Option<String>,
    held: Option<String>,
    held_error: Option<String>,
    nfs_refused: Option<String>,
    pods_refused: Option<String>,
    pinned: Vec<(String, Vec<(String, usize)>)>,
    followed: Option<String>,
    from_mark: u64,
}

impl State {
    fn inventory(&self) -> Inventory {
        Inventory {
            latest: self.latest.clone(),
            held: self.held.clone(),
            pinned: self.pinned.iter().map(|(p, _)| p.clone()).collect(),
            followed: self.followed.clone(),
        }
    }

    fn note_followed(&mut self, mirrored_at: Option<u64>) -> bool {
        let Some(released) = self.policy.as_ref().and_then(|p| p.released.as_ref()) else {
            return false;
        };
        let Some(from) = released.from.clone() else {
            return false;
        };
        let Some(latest) = &self.latest else {
            return false;
        };
        let elapsed = |p: &str| parse_prefix(p).map(|(n, _)| n);
        let listed_after = mirrored_at == Some(self.from_mark);
        let seen = *latest == from
            || (listed_after
                && released.is_runs(latest)
                && elapsed(latest).is_some_and(|n| Some(n) >= elapsed(&from)));
        if !seen || self.followed.as_ref() == Some(&from) {
            return false;
        }
        self.followed = Some(from);
        true
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Applied {
    pub version: u64,
    pub held: Option<String>,
    pub nfs_refused: Option<String>,
}

type Forget = Arc<dyn Fn(&str) + Send + Sync>;

pub struct Gateway {
    cfg: GatewayConfig,
    state: Mutex<State>,
    applying: std::sync::Mutex<()>,
    root_guard: Arc<Mutex<()>>,
    pins: Mutex<()>,
    saving: Mutex<()>,
    durable: Mutex<Option<Vec<u8>>>,
    wake_durable: tokio::sync::Notify,
    forget: Mutex<Option<Forget>>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn held_real(ledger: &Ledger, policy: &Policy) -> Result<Option<String>, String> {
    let Some(held) = &policy.held else {
        return Ok(None);
    };
    if let Some(real) = ledger.real_of(&held.prefix).filter(|r| is_real_prefix(r)) {
        return Ok(Some(real.to_string()));
    }
    let named = held.nfs_path.as_deref().and_then(|nfs| {
        let mut parts = nfs.split('/').filter(|p| !p.is_empty()).rev();
        let (run, top) = (parts.next()?, parts.next()?);
        let digits = top.strip_prefix("elapsed_samples_")?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some(format!(
            "elapsed_samples_{:0SEQ_WIDTH$}/{run}",
            digits.parse::<u64>().ok()?
        ))
        .filter(|real| is_real_prefix(real))
    });
    named.map(Some).ok_or_else(|| {
        format!(
            "held {} was never offered here, and its nfsPath {:?} names no elapsed_samples_<E>/<run> checkpoint",
            held.prefix, held.nfs_path
        )
    })
}

impl Gateway {
    pub fn open(cfg: GatewayConfig) -> Self {
        let gateway = Self {
            cfg,
            state: Mutex::new(State::default()),
            applying: std::sync::Mutex::new(()),
            root_guard: Arc::new(Mutex::new(())),
            pins: Mutex::new(()),
            saving: Mutex::new(()),
            durable: Mutex::new(None),
            wake_durable: tokio::sync::Notify::new(),
            forget: Mutex::new(None),
        };
        let saved = gateway.load_state();
        {
            let mut st = gateway.state.lock();
            if let Some(saved) = &saved {
                st.policy = saved
                    .get("policy")
                    .filter(|p| !p.is_null())
                    .and_then(|p| Policy::from_json(p).ok());
                st.ledger = saved
                    .get("ledger")
                    .map(Ledger::from_json)
                    .unwrap_or_default();
                st.latest = saved
                    .get("latest")
                    .and_then(Value::as_str)
                    .filter(|real| is_real_prefix(real))
                    .map(str::to_string);
                st.followed = saved
                    .get("followed")
                    .and_then(Value::as_str)
                    .filter(|real| is_real_prefix(real))
                    .map(str::to_string);
            }
        }
        let _ = fs::create_dir_all(gateway.held_root());
        remove_staging(&gateway.held_root());
        remove_temp_files(&gateway.cfg.root.join(GATEWAY_DIR));
        if let Some(policy) = gateway.policy() {
            gateway.pin_policy(&policy);
        }
        let st = gateway.state.lock();
        info!(
            "gateway: opened with policy {:?}, held {:?}, {} pinned",
            st.policy.as_ref().map(|p| (p.version, p.mode.as_str())),
            st.held,
            st.pinned.len()
        );
        drop(st);
        gateway
    }

    fn held_root(&self) -> PathBuf {
        self.cfg.root.join(GATEWAY_DIR).join(HELD_DIR)
    }

    fn state_copy_dir(&self) -> Option<PathBuf> {
        self.cfg
            .state_dir
            .as_ref()
            .map(|d| d.join(format!("s{}", self.cfg.server_index)))
    }

    fn load_state(&self) -> Option<Value> {
        let ours = |v: &Value| match &self.cfg.binding {
            Some(want) => v
                .pointer("/policy/binding")
                .and_then(binding_key)
                .is_some_and(|got| got == *want),
            None => true,
        };
        let read = |p: PathBuf| -> Option<Value> {
            let v: Value = serde_json::from_slice(&fs::read(p).ok()?).ok()?;
            ours(&v).then_some(v)
        };
        let local = read(self.cfg.root.join(GATEWAY_DIR).join(STATE_FILE));
        let durable = self.state_copy_dir().and_then(|d| read(d.join(STATE_FILE)));
        let version = |v: &Option<Value>| {
            v.as_ref()
                .and_then(|v| v.pointer("/policy/version"))
                .and_then(Value::as_u64)
        };
        if version(&durable) > version(&local) {
            durable
        } else {
            local.or(durable)
        }
    }

    fn state_bytes(&self) -> Vec<u8> {
        let st = self.state.lock();
        json!({
            "policy": st.policy.as_ref().map(|p| p.raw.clone()),
            "ledger": st.ledger.to_json(),
            "latest": st.latest,
            "followed": st.followed,
        })
        .to_string()
        .into_bytes()
    }

    fn save_state(&self) {
        let _one_at_a_time = self.saving.lock();
        let dir = self.cfg.root.join(GATEWAY_DIR);
        if let Err(e) = write_atomic(&dir, STATE_FILE, &self.state_bytes()) {
            warn!("gateway: saving state in {}: {e}", dir.display());
        }
        self.wake_durable.notify_one();
    }

    pub fn policy(&self) -> Option<Policy> {
        self.state.lock().policy.clone()
    }

    pub fn latest(&self) -> Option<String> {
        self.state.lock().latest.clone()
    }

    pub fn root_guard(&self) -> Arc<Mutex<()>> {
        self.root_guard.clone()
    }

    pub fn on_unpin(&self, forget: impl Fn(&str) + Send + Sync + 'static) {
        *self.forget.lock() = Some(Arc::new(forget));
    }

    pub fn note_live_pods(&self, live: HashMap<i64, HashSet<String>>) {
        self.state.lock().ledger.live = Some(
            live.into_iter()
                .map(|(slot, uids)| (slot, uids.into_iter().collect()))
                .collect(),
        );
    }

    pub fn note_bridge_pods(&self, ips: HashSet<IpAddr>) {
        self.state.lock().ledger.bridges = ips
            .into_iter()
            .map(|ip| Peer::Unknown(ip.to_string()).key())
            .collect();
    }

    pub fn identify_peers(&self, peers: &dyn PeerDirectory) {
        let ttl = self.cfg.unknown_peer_ttl.as_secs();
        let changed = self
            .state
            .lock()
            .ledger
            .identify(|ip| peers.lookup(ip), now_secs(), ttl);
        if changed {
            self.save_state();
        }
    }

    fn fresh(&self, now: Instant) -> bool {
        self.state
            .lock()
            .read_at
            .is_some_and(|at| now.saturating_duration_since(at) <= self.cfg.policy_fresh)
    }

    pub fn apply(&self, raw: &Value, now: Instant) -> Result<Applied, String> {
        let _one_at_a_time = self.applying.lock().unwrap_or_else(|e| e.into_inner());
        let policy = Policy::from_json(raw)?;
        if let (Some(want), Some(got)) = (&self.cfg.binding, policy.binding_key())
            && *want != got
        {
            return Err(format!(
                "the policy is for binding {got}; this gateway serves binding {want}"
            ));
        }
        {
            let mut st = self.state.lock();
            if let Some(current) = &st.policy {
                let (old_binding, new_binding) = (current.binding_key(), policy.binding_key());
                if old_binding.is_some() && new_binding.is_some() && old_binding != new_binding {
                    info!(
                        "gateway: binding {new_binding:?} replaces {old_binding:?}; starting a fresh ledger"
                    );
                    st.ledger = Ledger {
                        live: st.ledger.live.take(),
                        ..Ledger::default()
                    };
                } else if st.read_at.is_some() && policy.version < current.version {
                    return Err(format!(
                        "policy version {} is older than the applied {}",
                        policy.version, current.version
                    ));
                }
            }
            if st.policy.as_ref().map(|p| p.version) != Some(policy.version) {
                info!(
                    "gateway: policy version {} mode {}",
                    policy.version,
                    policy.mode.as_str()
                );
            }
            if held_real(&st.ledger, &policy).ok().flatten() != st.held {
                st.held = None;
            }
            let from = |p: Option<&Policy>| p.and_then(|p| p.released.as_ref()?.from.clone());
            if from(Some(&policy)) != from(st.policy.as_ref()) {
                st.from_mark += 1;
            }
            st.policy = Some(policy.clone());
            st.read_at = Some(now);
            st.note_followed(None);
        }
        let held = self.pin_policy(&policy);
        let nfs_refused = if self.cfg.server_index == 0 {
            let follow = (policy.mode == Mode::Follow)
                .then(|| candidate_offer(&policy, &self.state.lock().inventory()))
                .flatten();
            sync_serving_dir(&policy, follow.as_ref()).err()
        } else {
            None
        };
        if let Some(why) = &nfs_refused {
            warn!("gateway: NFS serving dir unchanged: {why}");
        }
        self.state.lock().nfs_refused = nfs_refused.clone();
        self.save_state();
        Ok(Applied {
            version: policy.version,
            held,
            nfs_refused,
        })
    }

    fn pin_policy(&self, policy: &Policy) -> Option<String> {
        let wanted = held_real(&self.state.lock().ledger, policy);
        let pinned = match &wanted {
            Ok(Some(real)) => self.pin_local(real).map(|()| real.clone()).map_err(|e| {
                warn!("gateway: pinning the held checkpoint: {e}");
                self.wake_durable.notify_one();
                e
            }),
            Ok(None) => Err(String::new()),
            Err(e) => Err(e.clone()),
        };
        let held = pinned.as_ref().ok().cloned();
        self.unpin_others(held.as_deref(), policy);
        let mut st = self.state.lock();
        st.held = held.clone();
        st.held_error = match (policy.held.is_some(), pinned) {
            (true, Err(e)) => Some(e),
            _ => None,
        };
        st.pinned = scan_prefixes(&self.held_root());
        held
    }

    fn pin_local(&self, real: &str) -> Result<(), String> {
        let _pins = self.pins.lock();
        let held_root = self.held_root();
        let _root = self.root_guard.lock();
        let src = self.cfg.root.join(real);
        let pinned = held_root.join(real);
        let src_files = is_real_dir(&self.cfg.root, real)
            .then(|| tree_files(&src))
            .transpose();
        if is_real_dir(&held_root, real) {
            match &src_files {
                Ok(Some(files)) if tree_files(&pinned).ok().as_ref() != Some(files) => {
                    warn!("gateway: the pinned {real} is not the serving root's; pinning it again");
                }
                _ => return Ok(()),
            }
        }
        let src_files =
            src_files.map_err(|e| format!("reading {real} in the serving root: {e}"))?;
        let Some(expected) = src_files else {
            return Err(format!(
                "the held {real} is in neither the held dir nor the serving root{}",
                if self.cfg.state_dir.is_some() {
                    "; seeding it from the state dir"
                } else {
                    ""
                }
            ));
        };
        if fs::symlink_metadata(&pinned).is_ok() {
            self.forget_pin(real);
        }
        stage_and_publish(&held_root, real, "link", None, true, |staging| {
            link_tree(&src, staging)?;
            check_tree(staging, &expected)
        })
        .map_err(|e| format!("pinning {real}: {e}"))?;
        info!("gateway: pinned {real}");
        Ok(())
    }

    fn forget_pin(&self, real: &str) {
        if let Some(forget) = self.forget.lock().clone() {
            forget(&format!("{GATEWAY_DIR}/{HELD_DIR}/{real}/"));
        }
    }

    fn unpin_others(&self, held: Option<&str>, policy: &Policy) {
        let keep_none = policy.mode == Mode::Follow && policy.held.is_none();
        if held.is_none() && !keep_none {
            return;
        }
        let _pins = self.pins.lock();
        let root = self.held_root();
        for (real, _) in scan_prefixes(&root) {
            if Some(real.as_str()) != held {
                self.forget_pin(&real);
                remove_prefix(&root, &real);
                info!("gateway: unpinned {real}");
            }
        }
    }

    pub fn reconcile_durable(&self) -> Result<(), String> {
        let Some(dir) = self.state_copy_dir() else {
            return Ok(());
        };
        let mut written = self.durable.lock();
        let copies = dir.join(HELD_DIR);
        let mut problems = Vec::new();
        remove_staging(&copies);
        remove_temp_files(&dir);
        if let Err(e) = self.seed_held(&copies) {
            problems.push(e);
        }
        let (policy, held) = {
            let st = self.state.lock();
            (st.policy.clone(), st.held.clone())
        };
        if let Some(real) = &held {
            let pin = self.held_root().join(real);
            match tree_files(&pin) {
                Err(e) => problems.push(format!("reading the pinned {real}: {e}")),
                Ok(files) => {
                    let copied = is_real_dir(&copies, real)
                        .then(|| tree_files(&copies.join(real)).ok())
                        .flatten();
                    if copied.as_ref() != Some(&files) {
                        match stage_and_publish(&copies, real, "copy", None, true, |staging| {
                            copy_tree(&pin, staging)?;
                            check_tree(staging, &files)
                        }) {
                            Ok(()) => {
                                info!("gateway: copied the held {real} to {}", copies.display())
                            }
                            Err(e) => problems.push(format!(
                                "copying the held {real} to {}: {e}",
                                copies.display()
                            )),
                        }
                    }
                }
            }
        }
        let keep_none = policy
            .as_ref()
            .is_some_and(|p| p.mode == Mode::Follow && p.held.is_none());
        if held.is_some() || keep_none {
            for (real, _) in scan_prefixes(&copies) {
                if Some(&real) != held.as_ref() {
                    remove_prefix(&copies, &real);
                }
            }
        }
        let bytes = self.state_bytes();
        if written.as_deref() != Some(bytes.as_slice()) {
            match write_atomic(&dir, STATE_FILE, &bytes) {
                Ok(()) => *written = Some(bytes),
                Err(e) => problems.push(format!("saving state in {}: {e}", dir.display())),
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(problems.join("; "))
        }
    }

    fn seed_held(&self, copies: &Path) -> Result<(), String> {
        let wanted = {
            let st = self.state.lock();
            if st.held.is_some() {
                return Ok(());
            }
            match st.policy.as_ref().map(|p| held_real(&st.ledger, p)) {
                Some(Ok(Some(real))) => real,
                _ => return Ok(()),
            }
        };
        if is_real_dir(&self.held_root(), &wanted) {
            self.adopt_held(&wanted);
            return Ok(());
        }
        if !is_real_dir(copies, &wanted) {
            return Ok(());
        }
        let from = copies.join(&wanted);
        let staged = stage_and_publish(
            &self.held_root(),
            &wanted,
            "seed",
            Some(&self.pins),
            false,
            |staging| {
                let files = tree_files(&from)?;
                copy_tree(&from, staging)?;
                check_tree(staging, &files)
            },
        );
        if let Err(e) = staged {
            let why = format!("seeding the held {wanted} from {}: {e}", copies.display());
            self.state.lock().held_error = Some(why.clone());
            return Err(why);
        }
        info!(
            "gateway: seeded the held {wanted} from {}",
            copies.display()
        );
        self.adopt_held(&wanted);
        Ok(())
    }

    fn adopt_held(&self, real: &str) {
        {
            let mut st = self.state.lock();
            let still_wanted = st
                .policy
                .as_ref()
                .and_then(|p| held_real(&st.ledger, p).ok().flatten());
            if st.held.is_none() && still_wanted.as_deref() == Some(real) {
                st.held = Some(real.to_string());
                st.held_error = None;
            }
            st.pinned = scan_prefixes(&self.held_root());
        }
        self.save_state();
    }

    pub fn probe_mark(&self) -> u64 {
        self.state.lock().from_mark
    }

    pub fn note_mirrored(&self, newest: &str) {
        self.note_mirrored_probed(newest, self.probe_mark());
    }

    pub fn note_mirrored_probed(&self, newest: &str, mark: u64) {
        let Some((_, real)) = parse_prefix(newest).filter(|(_, real)| is_real_prefix(real)) else {
            return;
        };
        if !self.cfg.root.join(&real).is_dir() {
            return;
        }
        let (moved, followed) = {
            let mut st = self.state.lock();
            let moved = st.latest.as_deref() != Some(real.as_str());
            st.latest = Some(real.clone());
            (moved, st.note_followed(Some(mark)))
        };
        if moved {
            info!("gateway: the trainer's newest mirrored checkpoint is {real}");
        }
        if moved || followed {
            self.save_state();
        }
    }

    pub fn rewind_to(&self, newest: &str) -> bool {
        let Some((_, real)) = parse_prefix(newest) else {
            return false;
        };
        let top = real.split('/').next().expect("prefix has a top");
        let _root = self.root_guard.lock();
        let Ok(entries) = fs::read_dir(&self.cfg.root) else {
            return false;
        };
        let other_run_at_top = !self.cfg.root.join(&real).is_dir();
        let mut dropped = false;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let above = name.as_str() > top || (name == top && other_run_at_top);
            if name.starts_with("elapsed_samples_") && above && entry.path().is_dir() {
                info!("gateway: the trainer's newest is {real}; dropping {name}");
                let _ = fs::remove_dir_all(entry.path());
                dropped = true;
            }
        }
        if dropped {
            let mut st = self.state.lock();
            if st
                .latest
                .as_ref()
                .is_some_and(|l| !self.cfg.root.join(l).is_dir())
            {
                st.latest = None;
            }
        }
        dropped
    }

    pub fn mirrors(&self, newest: &str) -> bool {
        parse_prefix(newest).is_some_and(|(_, real)| self.cfg.root.join(real).is_dir())
    }

    fn peer(&self, ip: Option<IpAddr>, peers: &dyn PeerDirectory) -> Peer {
        match ip.and_then(|ip| peers.lookup(ip).map(|p| (ip, p))) {
            Some((_, (slot_id, uid))) => Peer::Pod { slot_id, uid },
            None => Peer::Unknown(ip.map_or_else(|| "unknown".to_string(), |ip| ip.to_string())),
        }
    }

    pub fn decide(&self, peer: &Peer, now: Instant) -> Option<Offer> {
        let fresh = self.fresh(now);
        let mut st = self.state.lock();
        let inv = st.inventory();
        let policy = st.policy.clone();
        let how = Decide {
            fresh,
            may_admit: self.cfg.server_index == 0,
            now_secs: now_secs(),
        };
        st.ledger.decide(policy.as_ref(), peer, &inv, how)
    }

    pub fn list_for(
        &self,
        peer: &Peer,
        upstream: &[(String, usize)],
        now: Instant,
    ) -> Vec<(String, usize)> {
        let Some(offer) = self.decide(peer, now) else {
            return Vec::new();
        };
        let rename = |rest: &str| format!("{}/{rest}", offer.serving);
        match &offer.source {
            Source::Upstream(real) => {
                let dir = format!("{real}/");
                upstream
                    .iter()
                    .filter_map(|(name, size)| Some((rename(name.strip_prefix(&dir)?), *size)))
                    .collect()
            }
            Source::Held(real) => self
                .state
                .lock()
                .pinned
                .iter()
                .find(|(p, _)| p == real)
                .map(|(_, files)| files.iter().map(|(f, s)| (rename(f), *s)).collect())
                .unwrap_or_default(),
        }
    }

    pub fn map_send(&self, peer: &Peer, names: &[String]) -> Result<Vec<String>, Status> {
        let now = now_secs();
        let mut st = self.state.lock();
        names
            .iter()
            .map(|name| {
                let serving = prefix_of(name).ok_or_else(|| {
                    Status::invalid_argument(format!("{name} has no checkpoint prefix"))
                })?;
                let rest = &name[serving.len() + 1..];
                if rest
                    .split('/')
                    .any(|c| c.is_empty() || c == "." || c == "..")
                {
                    return Err(Status::invalid_argument(format!(
                        "{name} is not a checkpoint file"
                    )));
                }
                match st.ledger.allowed(peer, serving, now) {
                    Some(Source::Upstream(real)) if is_real_prefix(&real) => {
                        Ok(format!("{real}/{rest}"))
                    }
                    Some(Source::Held(real)) if is_real_prefix(&real) => {
                        Ok(format!("{GATEWAY_DIR}/{HELD_DIR}/{real}/{rest}"))
                    }
                    _ => Err(Status::not_found(format!(
                        "{serving} was not offered to this pod by the prod checkpoint gateway"
                    ))),
                }
            })
            .collect()
    }

    fn note_download(&self, peer: &Peer, prefix: &str, streamed: u64, rdma: u64, ok: bool) {
        self.state
            .lock()
            .ledger
            .note_download(peer, prefix, streamed, rdma, ok, now_secs());
    }

    pub fn note_pods_refused(&self, why: Option<String>) {
        self.state.lock().pods_refused = why;
    }

    pub fn ack(&self) -> Option<Value> {
        let ttl = self.cfg.unknown_peer_ttl.as_secs();
        let mut st = self.state.lock();
        if st.ledger.forget_unknown(now_secs(), ttl) {
            drop(st);
            self.save_state();
            st = self.state.lock();
        }
        let policy = st.policy.as_ref()?;
        let mut body = st.ledger.ack();
        body["version"] = json!(policy.version);
        body["gateway"] = json!(self.cfg.name);
        body["heldPinned"] = match policy.held {
            Some(_) => json!(st.held.is_some()),
            None => Value::Null,
        };
        body["heldError"] = json!(st.held_error);
        body["nfsRefused"] = json!(st.nfs_refused);
        body["podsRefused"] = json!(st.pods_refused);
        Some(body)
    }
}

fn is_real_dir(root: &Path, real: &str) -> bool {
    let Some((top, child)) = real.split_once('/') else {
        return false;
    };
    let dir = |p: &Path| fs::symlink_metadata(p).is_ok_and(|m| m.is_dir());
    dir(&root.join(top)) && dir(&root.join(top).join(child))
}

fn scan_prefixes(root: &Path) -> Vec<(String, Vec<(String, usize)>)> {
    let Ok(tops) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for top in tops.flatten() {
        let top_name = top.file_name().to_string_lossy().into_owned();
        if !top_name.starts_with("elapsed_samples_") || !top.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let Ok(children) = fs::read_dir(top.path()) else {
            continue;
        };
        for child in children.flatten() {
            let child_name = child.file_name().to_string_lossy().into_owned();
            if child_name.starts_with('.') || !child.file_type().is_ok_and(|t| t.is_dir()) {
                continue;
            }
            let mut files = Vec::new();
            walk_files(&child.path(), "", &mut files);
            files.sort();
            out.push((format!("{top_name}/{child_name}"), files));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn walk_files(dir: &Path, rel: &str, out: &mut Vec<(String, usize)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel_name = if rel.is_empty() {
            name
        } else {
            format!("{rel}/{name}")
        };
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            walk_files(&entry.path(), &rel_name, out);
        } else if kind.is_file()
            && let Ok(meta) = entry.metadata()
        {
            out.push((rel_name, meta.len() as usize));
        }
    }
}

fn not_a_checkpoint_file(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "{} is not a regular file or directory; a checkpoint holds only those",
            path.display()
        ),
    )
}

fn tree_files(dir: &Path) -> io::Result<Vec<(String, usize)>> {
    fn walk(dir: &Path, rel: &str, out: &mut Vec<(String, usize)>) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let rel_name = if rel.is_empty() {
                name
            } else {
                format!("{rel}/{name}")
            };
            let kind = entry.file_type()?;
            if kind.is_dir() {
                walk(&entry.path(), &rel_name, out)?;
            } else if kind.is_file() {
                out.push((rel_name, entry.metadata()?.len() as usize));
            } else {
                return Err(not_a_checkpoint_file(&entry.path()));
            }
        }
        Ok(())
    }
    if !fs::symlink_metadata(dir)?.is_dir() {
        return Err(not_a_checkpoint_file(dir));
    }
    let mut out = Vec::new();
    walk(dir, "", &mut out)?;
    out.sort();
    Ok(out)
}

fn check_tree(staged: &Path, expected: &[(String, usize)]) -> io::Result<()> {
    let got = tree_files(staged)?;
    if got != expected {
        return Err(io::Error::other(format!(
            "staged {} of {} files, or sizes differ: the source changed while it was read",
            got.len(),
            expected.len()
        )));
    }
    Ok(())
}

fn remove_prefix(root: &Path, real: &str) {
    let _ = fs::remove_dir_all(root.join(real));
    if let Some((top, _)) = real.split_once('/') {
        let _ = fs::remove_dir(root.join(top));
    }
}

fn remove_staging(root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with('.') {
            let path = entry.path();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                let _ = fs::remove_dir_all(path);
            } else {
                let _ = fs::remove_file(path);
            }
        }
    }
}

fn stage_and_publish(
    root: &Path,
    real: &str,
    tag: &str,
    publish_lock: Option<&Mutex<()>>,
    replace: bool,
    fill: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<()> {
    let (top, child) = real.split_once('/').expect("a real prefix is top/child");
    fs::create_dir_all(root)?;
    let staging = root.join(format!(".{top}.{child}.{tag}"));
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir(&staging)?;
    if let Err(e) = fill(&staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(e);
    }
    let _publishing = publish_lock.map(Mutex::lock);
    let published = (|| {
        let top_dir = root.join(top);
        if fs::symlink_metadata(&top_dir).is_ok_and(|m| !m.is_dir()) {
            fs::remove_file(&top_dir)?;
        }
        fs::create_dir_all(&top_dir)?;
        let target = top_dir.join(child);
        match fs::symlink_metadata(&target) {
            Ok(m) if m.is_dir() && !replace => return fs::remove_dir_all(&staging),
            Ok(m) if m.is_dir() => fs::remove_dir_all(&target)?,
            Ok(_) => fs::remove_file(&target)?,
            Err(_) => {}
        }
        fs::rename(&staging, &target)
    })();
    if published.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    published
}

fn stage_and_rename(
    root: &Path,
    top: &str,
    fill: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<()> {
    fs::create_dir_all(root)?;
    let staging = root.join(format!(".{top}"));
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    if let Err(e) = fill(&staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err(e);
    }
    let target = root.join(top);
    let _ = fs::remove_dir_all(&target);
    fs::rename(&staging, &target)
}

fn link_tree(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        let kind = entry.file_type()?;
        if kind.is_dir() {
            link_tree(&src, &dst)?;
        } else if kind.is_file() {
            fs::hard_link(&src, &dst)?;
        } else {
            return Err(not_a_checkpoint_file(&src));
        }
    }
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_tree(&src, &dst)?;
        } else if kind.is_file() {
            let mut input = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&src)?;
            if !input.metadata()?.is_file() {
                return Err(not_a_checkpoint_file(&src));
            }
            let mut output = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&dst)?;
            io::copy(&mut input, &mut output)?;
        } else {
            return Err(not_a_checkpoint_file(&src));
        }
    }
    Ok(())
}

fn write_atomic(dir: &Path, name: &str, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let written = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .and_then(|mut f| f.write_all(bytes))
        .and_then(|()| fs::rename(&tmp, dir.join(name)));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written
}

fn remove_temp_files(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(&format!(".{STATE_FILE}.")) && name.ends_with(".tmp") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

fn sync_serving_dir(policy: &Policy, follow: Option<&Offer>) -> Result<(), String> {
    let Some(nfs) = &policy.nfs else {
        return Ok(());
    };
    if let Some(offer) = follow.filter(|o| {
        nfs.offered
            .as_deref()
            .and_then(parse_prefix)
            .is_none_or(|(seq, _)| o.seq > seq)
    }) {
        return follow_serving_dir(policy, nfs, offer);
    }
    let Some(offered) = &nfs.offered else {
        return Ok(());
    };
    let (seq, offered) = parse_prefix(offered)
        .ok_or_else(|| format!("nfs.offered {offered:?} is not a checkpoint prefix"))?;
    if let Some(limit) = policy.nfs_seq_limit()
        && seq > limit
    {
        return Err(format!(
            "nfs.offered {offered} sorts above the held serving sequence {limit} while pods are held"
        ));
    }
    let leaf = match (&policy.held, &policy.released, nfs.offered_path.as_deref()) {
        (Some(held), _, _) if held.prefix == offered => held
            .nfs_path
            .as_deref()
            .ok_or_else(|| format!("held {offered} names no nfsPath"))?,
        (_, Some(released), Some(path)) if offered.ends_with(&format!("/{}", released.key())) => {
            let checkpoint = i128::from(seq) - i128::from(released.seq_offset);
            let mut parts = path.trim_end_matches('/').rsplit('/');
            let (run, top) = (parts.next(), parts.next());
            let elapsed = top
                .and_then(|t| t.strip_prefix("elapsed_samples_"))
                .filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|n| n.parse::<i128>().ok());
            if !path.starts_with('/')
                || !run.is_some_and(|run| same_run(run, &released.run))
                || elapsed != Some(checkpoint)
            {
                return Err(format!(
                    "nfs.offeredPath {path} is not the released checkpoint of {offered} \
                     (…/elapsed_samples_{checkpoint}/{})",
                    released.run
                ));
            }
            path
        }
        _ => return Err(format!("no NFS checkpoint is known for {offered}")),
    };
    write_serving_dir(Path::new(&nfs.dir), &offered, seq, Path::new(leaf))
        .map_err(|e| format!("writing {} for {offered}: {e}", nfs.dir))
}

fn follow_serving_dir(policy: &Policy, nfs: &NfsPolicy, offer: &Offer) -> Result<(), String> {
    let Source::Upstream(real) = &offer.source else {
        return Ok(());
    };
    let Some(named) = nfs
        .offered_path
        .as_deref()
        .or_else(|| policy.held.as_ref()?.nfs_path.as_deref())
    else {
        return Ok(());
    };
    let job = Path::new(named.trim_end_matches('/'))
        .parent()
        .and_then(Path::parent)
        .filter(|j| j.is_absolute())
        .ok_or_else(|| format!("{named} is not …/elapsed_samples_<E>/<run>"))?;
    let (elapsed, prefix) =
        parse_prefix(real).ok_or_else(|| format!("{real} is not a checkpoint prefix"))?;
    let (top, run) = prefix.split_once('/').expect("parse_prefix gives top/name");
    let leaf = [
        top.to_string(),
        format!("elapsed_samples_{elapsed}"),
        format!("elapsed_samples_{elapsed:0SEQ_WIDTH$}"),
    ]
    .into_iter()
    .map(|t| job.join(t).join(run))
    .find(|l| l.is_dir())
    .ok_or_else(|| {
        format!(
            "{} (pods are offered {}) is not on NFS yet",
            job.join(format!("elapsed_samples_{elapsed}"))
                .join(run)
                .display(),
            offer.serving
        )
    })?;
    if let Some(why) = unfinished_save(&leaf) {
        return Err(format!(
            "{} (pods are offered {}) is still being written on NFS: {why}",
            leaf.display(),
            offer.serving
        ));
    }
    write_serving_dir(Path::new(&nfs.dir), &offer.serving, offer.seq, &leaf)
        .map_err(|e| format!("writing {} for {}: {e}", nfs.dir, offer.serving))
}

fn unfinished_save(leaf: &Path) -> Option<&'static str> {
    let Ok(entries) = fs::read_dir(leaf) else {
        return Some("its listing failed");
    };
    let names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if names.iter().any(|n| n.contains(".orbax-checkpoint-tmp")) {
        return Some("its orbax tmp dir is not renamed yet");
    }
    let finished = leaf.join("completed").is_file()
        || leaf.join(".COMPLETED").is_file()
        || leaf.join("orbax-ckpt").is_dir();
    (!finished).then_some("it has no `completed` yet")
}

fn write_serving_dir(dir: &Path, offered: &str, seq: u64, leaf: &Path) -> io::Result<()> {
    let (top, name) = offered
        .split_once('/')
        .expect("parse_prefix gives top/name");
    if !leaf.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("checkpoint {} is not a directory", leaf.display()),
        ));
    }
    fs::create_dir_all(dir)?;
    let entries: Vec<(String, u64)> = fs::read_dir(dir)?
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            let (s, _) = parse_prefix(&format!("{n}/x"))?;
            Some((n, s))
        })
        .collect();
    for (n, s) in &entries {
        if *s > seq {
            fs::remove_dir_all(dir.join(n))?;
        }
    }
    let current = dir.join(top).join(name);
    let up_to_date = current.is_dir()
        && fs::read_dir(leaf)?
            .flatten()
            .all(|c| fs::read_link(current.join(c.file_name())).ok() == Some(c.path()));
    if !up_to_date {
        stage_and_rename(dir, top, |staging| {
            let leaf_dir = staging.join(name);
            fs::create_dir_all(&leaf_dir)?;
            for child in fs::read_dir(leaf)? {
                let child = child?;
                std::os::unix::fs::symlink(child.path(), leaf_dir.join(child.file_name()))?;
            }
            Ok(())
        })?;
        info!(
            "gateway: NFS serving dir {} now offers {offered}",
            dir.display()
        );
    }
    for (n, s) in &entries {
        if *s < seq {
            fs::remove_dir_all(dir.join(n))?;
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct GatewayService<S> {
    inner: S,
    gateway: Arc<Gateway>,
    peers: Arc<dyn PeerDirectory>,
}

impl<S> GatewayService<S> {
    pub fn new(inner: S, gateway: Arc<Gateway>, peers: Arc<dyn PeerDirectory>) -> Self {
        Self {
            inner,
            gateway,
            peers,
        }
    }
}

impl<S> NamedService for GatewayService<S> {
    const NAME: &'static str = "copy.Copy";
}

type Inner = http::Request<body::Body>;

impl<S> Service<Inner> for GatewayService<S>
where
    S: Service<Inner, Response = http::Response<body::Body>, Error = Infallible>
        + Clone
        + Send
        + 'static,
    S::Future: Send,
{
    type Response = http::Response<body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Inner) -> Self::Future {
        let mut inner = self.inner.clone();
        let gateway = self.gateway.clone();
        let peers = self.peers.clone();
        Box::pin(async move {
            let peer = gateway.peer(peer_ip(&request), peers.as_ref());
            let path = request.uri().path().to_string();
            Ok(match path.as_str() {
                LIST => match list(&mut inner, &gateway, &peer, request).await {
                    Ok(response) => response,
                    Err(status) => make_response(Err(status)),
                },
                SEND => match send(&mut inner, gateway.clone(), &peer, request).await {
                    Ok(response) => response,
                    Err(status) => make_response(Err(status)),
                },
                REGISTER => make_response(Err(Status::unimplemented(
                    "the prod checkpoint gateway serves only List and Send",
                ))),
                _ => make_response(Err(Status::unimplemented(format!(
                    "{path} not implemented"
                )))),
            })
        })
    }
}

async fn call_inner<S>(inner: &mut S, request: Inner) -> http::Response<body::Body>
where
    S: Service<Inner, Response = http::Response<body::Body>, Error = Infallible>,
{
    let _ = futures::future::poll_fn(|cx| inner.poll_ready(cx)).await;
    match inner.call(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    }
}

async fn list<S>(
    inner: &mut S,
    gateway: &Gateway,
    peer: &Peer,
    request: Inner,
) -> Result<http::Response<body::Body>, Status>
where
    S: Service<Inner, Response = http::Response<body::Body>, Error = Infallible>,
{
    let response = call_inner(inner, request).await;
    if let Some(status) = Status::from_header_map(response.headers())
        && status.code() != tonic::Code::Ok
    {
        return Ok(response);
    }
    let (mut all_names, mut prefix_sizes, mut suffix_sizes, mut sizes) = (
        Vec::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
    );
    parse(
        proto![
            (1, bytes(&mut all_names)),
            (2, repeated_ints(&mut prefix_sizes)),
            (3, repeated_ints(&mut suffix_sizes)),
            (4, repeated_ints(&mut sizes)),
        ],
        response.into_body(),
        BodyKind::Response,
    )
    .await?;
    let upstream: Vec<(String, usize)> = decode_names(all_names, prefix_sizes, suffix_sizes)?
        .into_iter()
        .map(|n| String::from_utf8_lossy(&n).into_owned())
        .zip(sizes)
        .collect();
    let mut listed = gateway.list_for(peer, &upstream, Instant::now());
    listed.sort();
    let (names, sizes): (Vec<_>, Vec<_>) =
        listed.into_iter().map(|(n, s)| (n.into_bytes(), s)).unzip();
    let (all_names, prefix_sizes, suffix_sizes) = encode_names(names);
    let mut buf = get_bytes_mut();
    buf.put_string(1, &all_names);
    buf.put_repeated_ints([2, 3, 4], [&prefix_sizes, &suffix_sizes, &sizes]);
    Ok(make_response(Ok(add_ok_trailer(freeze(buf)))))
}

async fn send<S>(
    inner: &mut S,
    gateway: Arc<Gateway>,
    peer: &Peer,
    request: Inner,
) -> Result<http::Response<body::Body>, Status>
where
    S: Service<Inner, Response = http::Response<body::Body>, Error = Infallible>,
{
    let (parts, request_body) = request.into_parts();
    let (mut names, mut prefix_sizes, mut suffix_sizes) = (Vec::new(), Vec::new(), Vec::new());
    let (mut offsets, mut sizes) = (Vec::<usize>::new(), Vec::<usize>::new());
    let (mut endpoints, mut rmrs, mut prio) = (Vec::new(), Vec::new(), 0usize);
    parse(
        proto![
            (1, bytes(&mut names)),
            (2, repeated_ints(&mut prefix_sizes)),
            (3, repeated_ints(&mut suffix_sizes)),
            (4, repeated_ints(&mut offsets)),
            (5, repeated_ints(&mut sizes)),
            (6, repeated_bytes(&mut endpoints)),
            (7, repeated_bytes(&mut rmrs)),
            (8, |x| prio = x),
        ],
        request_body,
        BodyKind::Request,
    )
    .await?;
    let asked: Vec<String> = decode_names(names, prefix_sizes, suffix_sizes)?
        .into_iter()
        .map(String::from_utf8)
        .collect::<Result<_, _>>()
        .map_err(|e| Status::invalid_argument(format!("file names must be UTF-8: {e}")))?;
    let real = gateway.map_send(peer, &asked)?;
    let (all_names, prefix_sizes, suffix_sizes) =
        encode_names(real.into_iter().map(String::into_bytes).collect());
    let mut buf = get_bytes_mut();
    buf.put_string(1, &all_names);
    buf.put_repeated_ints(
        [2, 3, 4, 5],
        [&prefix_sizes, &suffix_sizes, &offsets, &sizes],
    );
    for endpoint in &endpoints {
        buf.put_string(6, endpoint);
    }
    for rmr in &rmrs {
        buf.put_string(7, rmr);
    }
    if prio != 0 {
        buf.put_varints([8 << 3, prio]);
    }
    let request = http::Request::from_parts(parts, body::Body::new(freeze(buf)));
    let response = call_inner(inner, request).await;
    let Some(prefix) = asked.first().and_then(|n| prefix_of(n)).map(str::to_string) else {
        return Ok(response);
    };
    let asked_bytes: u64 = sizes.iter().map(|&s| s as u64).sum();
    let rdma = !endpoints.is_empty();
    if Status::from_header_map(response.headers()).is_some_and(|s| s.code() != tonic::Code::Ok) {
        gateway.note_download(peer, &prefix, 0, 0, false);
        return Ok(response);
    }
    let peer = peer.clone();
    let report = Box::new(move |streamed: u64, ok: bool| {
        let (streamed, rdma_requested) = match rdma {
            true => (0, if ok { asked_bytes } else { 0 }),
            false => (streamed.min(asked_bytes), 0),
        };
        gateway.note_download(&peer, &prefix, streamed, rdma_requested, ok);
    });
    Ok(response.map(|inner| {
        body::Body::new(Observed {
            inner,
            streamed: 0,
            report: Some(report),
        })
    }))
}

type Report = Box<dyn FnOnce(u64, bool) + Send>;

struct Observed {
    inner: body::Body,
    streamed: u64,
    report: Option<Report>,
}

impl Observed {
    fn finish(&mut self, ok: bool) {
        if let Some(report) = self.report.take() {
            report(self.streamed, ok);
        }
    }
}

impl http_body::Body for Observed {
    type Data = bytes::Bytes;
    type Error = Status;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = &mut *self;
        let polled = std::pin::Pin::new(&mut this.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.streamed += data.len() as u64;
                }
                if let Some(trailers) = frame.trailers_ref() {
                    let ok = Status::from_header_map(trailers)
                        .is_none_or(|s| s.code() == tonic::Code::Ok);
                    this.finish(ok);
                }
            }
            Poll::Ready(Some(Err(_)) | None) => this.finish(false),
            Poll::Pending => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for Observed {
    fn drop(&mut self) {
        self.finish(false);
    }
}

struct JsonClient {
    channel: tonic::transport::Channel,
}

fn split_url(url: &str) -> io::Result<(String, String)> {
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidInput, why);
    let uri: http::Uri = url
        .parse()
        .map_err(|e| bad(format!("bad url {url}: {e}")))?;
    let (Some(scheme), Some(authority)) = (uri.scheme_str(), uri.authority()) else {
        return Err(bad(format!("url {url} must be absolute")));
    };
    Ok((
        format!("{scheme}://{authority}"),
        uri.path().trim_end_matches('/').to_string(),
    ))
}

impl JsonClient {
    fn new(origin: &str, tls: Option<&crate::tls::ClientTlsOptions>) -> io::Result<Self> {
        let endpoint = crate::tls::endpoint_for(origin, tls)?
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30));
        Ok(Self {
            channel: endpoint.connect_lazy(),
        })
    }

    async fn call(
        &self,
        method: http::Method,
        path: &str,
        bearer: Option<&str>,
        body: Option<Value>,
    ) -> Result<Value, String> {
        let mut builder = http::Request::builder()
            .method(method)
            .uri(path)
            .version(http::Version::HTTP_2)
            .header(http::header::ACCEPT, "application/json");
        if let Some(token) = bearer {
            let mut value = http::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| format!("{path}: the bearer token is not a valid header value"))?;
            value.set_sensitive(true);
            builder = builder.header(http::header::AUTHORIZATION, value);
        }
        let body = match body {
            Some(v) => {
                builder = builder.header(http::header::CONTENT_TYPE, "application/json");
                body::Body::new(http_body_util::Full::new(bytes::Bytes::from(v.to_string())))
            }
            None => body::Body::empty(),
        };
        let request = builder.body(body).map_err(|e| e.to_string())?;
        let mut channel = self.channel.clone();
        futures::future::poll_fn(|cx| channel.poll_ready(cx))
            .await
            .map_err(|e| format!("{path}: {e}"))?;
        let response = channel
            .call(request)
            .await
            .map_err(|e| format!("{path}: {e}"))?;
        let status = response.status();
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .map_err(|e| format!("{path}: {e}"))?
            .to_bytes();
        if !status.is_success() {
            return Err(format!(
                "{path} answered {status}: {}",
                String::from_utf8_lossy(&bytes[..bytes.len().min(512)])
            ));
        }
        if bytes.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))
    }
}

pub struct PolicyClient {
    path: String,
    http: JsonClient,
}

impl PolicyClient {
    pub fn new(url: &str, tls: Option<&crate::tls::ClientTlsOptions>) -> io::Result<Self> {
        let (origin, path) = split_url(url)?;
        Ok(Self {
            path,
            http: JsonClient::new(&origin, tls)?,
        })
    }

    pub async fn fetch(&self) -> Result<Value, String> {
        self.http
            .call(http::Method::GET, &self.path, None, None)
            .await
    }

    pub async fn ack(&self, body: Value) -> Result<(), String> {
        let path = format!("{}/ack", self.path);
        self.http
            .call(http::Method::POST, &path, None, Some(body))
            .await
            .map(|_| ())
    }
}

fn label_selector(selector: &Value) -> Option<String> {
    let labels = match selector {
        Value::String(s) => return Some(s.trim().to_string()).filter(|s| !s.is_empty()),
        Value::Object(o) if o.contains_key("matchExpressions") => return None,
        Value::Object(o) => match o.get("matchLabels") {
            Some(Value::Object(m)) => m,
            Some(_) => return None,
            None => o,
        },
        _ => return None,
    };
    let mut pairs = labels
        .iter()
        .map(|(k, v)| Some(format!("{k}={}", v.as_str()?)))
        .collect::<Option<Vec<_>>>()?;
    pairs.sort();
    Some(pairs.join(",")).filter(|s| !s.is_empty())
}

fn pods_of(list: &Value) -> Vec<(IpAddr, String)> {
    let mut out = Vec::new();
    for item in list["items"].as_array().into_iter().flatten() {
        let Some(uid) = item["metadata"]["uid"].as_str() else {
            continue;
        };
        let status = &item["status"];
        if matches!(status["phase"].as_str(), Some("Succeeded" | "Failed")) {
            continue;
        }
        let ips = status["podIPs"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|i| &i["ip"])
            .chain([&status["podIP"]])
            .filter_map(|ip| ip.as_str()?.parse::<IpAddr>().ok());
        for ip in ips {
            if !out.contains(&(ip, uid.to_string())) {
                out.push((ip, uid.to_string()));
            }
        }
    }
    out
}

const SERVICE_ACCOUNT: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

struct Listed {
    at: Instant,
    pods: Vec<(IpAddr, String)>,
}

pub struct KubePods {
    http: JsonClient,
    token: Option<PathBuf>,
    max_age: Duration,
    slots: Mutex<HashMap<i64, Listed>>,
    bridge_slots: Mutex<HashMap<i64, Listed>>,
    pods: parking_lot::RwLock<HashMap<IpAddr, (i64, String)>>,
    bridges: parking_lot::RwLock<HashSet<IpAddr>>,
    refused: parking_lot::RwLock<Option<String>>,
}

impl KubePods {
    pub fn in_cluster() -> io::Result<Self> {
        let host = std::env::var("KUBERNETES_SERVICE_HOST").map_err(|_| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "not in a Kubernetes pod (no KUBERNETES_SERVICE_HOST)",
            )
        })?;
        let port = std::env::var("KUBERNETES_SERVICE_PORT").unwrap_or_else(|_| "443".into());
        let host = if host.contains(':') {
            format!("[{host}]")
        } else {
            host
        };
        let account = Path::new(SERVICE_ACCOUNT);
        let tls = crate::tls::ClientTlsOptions {
            ca_path: account.join("ca.crt").to_string_lossy().into_owned(),
            cert_path: None,
            key_path: None,
            server_name: Some("kubernetes.default.svc".into()),
        };
        Self::new(
            &format!("https://{host}:{port}"),
            Some(&tls),
            Some(account.join("token")),
        )
    }

    pub fn new(
        origin: &str,
        tls: Option<&crate::tls::ClientTlsOptions>,
        token: Option<PathBuf>,
    ) -> io::Result<Self> {
        Ok(Self {
            http: JsonClient::new(origin, tls)?,
            token,
            max_age: Duration::from_secs(POD_STALE_SECS),
            slots: Default::default(),
            bridge_slots: Default::default(),
            pods: Default::default(),
            bridges: Default::default(),
            refused: Default::default(),
        })
    }

    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = max_age;
        self
    }

    pub async fn refresh(&self, policy: &Policy) -> Result<usize, String> {
        self.refresh_at(policy, Instant::now()).await
    }

    pub async fn refresh_at(&self, policy: &Policy, now: Instant) -> Result<usize, String> {
        let mut errors = Vec::new();
        let mut token_error = None;
        let mut slot_errors: Vec<(i64, String)> = Vec::new();
        let token = match &self.token {
            Some(path) => match fs::read_to_string(path) {
                Ok(token) => Some(token.trim().to_string()),
                Err(e) => {
                    let why = format!("reading the service account token: {e}");
                    errors.push(why.clone());
                    token_error = Some(why);
                    None
                }
            },
            None => None,
        };
        let mut listed = HashMap::new();
        let mut unusable = Vec::new();
        for slot in &policy.slots {
            if self.token.is_some() && token.is_none() {
                break;
            }
            let Some(selector) = label_selector(&slot.pod_selector) else {
                slot_errors.push((
                    slot.slot_id,
                    format!(
                        "slot {}: podSelector {} is not a non-empty label selector",
                        slot.slot_id, slot.pod_selector
                    ),
                ));
                unusable.push(slot.slot_id);
                continue;
            };
            if slot.namespace.is_empty() {
                slot_errors.push((slot.slot_id, format!("slot {}: no namespace", slot.slot_id)));
                unusable.push(slot.slot_id);
                continue;
            }
            match self
                .list(&slot.namespace, &selector, token.as_deref())
                .await
            {
                Ok(pods) => {
                    listed.insert(slot.slot_id, pods);
                }
                Err(e) => slot_errors.push((slot.slot_id, format!("slot {}: {e}", slot.slot_id))),
            }
        }
        errors.extend(slot_errors.iter().map(|(_, why)| why.clone()));
        let mut bridges_listed = HashMap::new();
        let mut bridges_unusable = Vec::new();
        for (slot_id, bridge) in policy
            .slots
            .iter()
            .filter_map(|s| Some((s.slot_id, s.bridge.as_ref()?)))
        {
            if self.token.is_some() && token.is_none() {
                break;
            }
            let selector =
                label_selector(&bridge.pod_selector).filter(|_| !bridge.namespace.is_empty());
            let Some(selector) = selector else {
                errors.push(format!(
                    "slot {slot_id}: bridge {} in namespace {:?} is not a non-empty label \
                     selector in a namespace",
                    bridge.pod_selector, bridge.namespace
                ));
                bridges_unusable.push(slot_id);
                continue;
            };
            match self
                .list(&bridge.namespace, &selector, token.as_deref())
                .await
            {
                Ok(pods) => {
                    bridges_listed.insert(slot_id, pods);
                }
                Err(e) => errors.push(format!("slot {slot_id}'s bridge: {e}")),
            }
        }
        {
            let mut slots = self.bridge_slots.lock();
            slots.retain(|id, listing| {
                policy
                    .slots
                    .iter()
                    .any(|s| s.slot_id == *id && s.bridge.is_some())
                    && !bridges_unusable.contains(id)
                    && now.saturating_duration_since(listing.at) <= self.max_age
            });
            for (id, pods) in bridges_listed {
                slots.insert(id, Listed { at: now, pods });
            }
        }
        let n = {
            let mut slots = self.slots.lock();
            slots.retain(|id, listing| {
                policy.slots.iter().any(|s| s.slot_id == *id)
                    && !unusable.contains(id)
                    && now.saturating_duration_since(listing.at) <= self.max_age
            });
            for (id, pods) in listed {
                slots.insert(id, Listed { at: now, pods });
            }
            let unlisted = |id: &i64| !slots.contains_key(id);
            let refused: Vec<String> = token_error
                .iter()
                .filter(|_| policy.slots.iter().any(|s| unlisted(&s.slot_id)))
                .cloned()
                .chain(
                    slot_errors
                        .iter()
                        .filter(|(id, _)| unlisted(id))
                        .map(|(_, why)| why.clone()),
                )
                .collect();
            *self.refused.write() = (!refused.is_empty()).then(|| refused.join("; "));
            let mut ids: Vec<_> = slots.keys().copied().collect();
            ids.sort();
            let mut pods: HashMap<IpAddr, (i64, String)> = HashMap::new();
            let mut shared = Vec::new();
            for id in ids {
                for (ip, uid) in &slots[&id].pods {
                    let pod = (id, uid.clone());
                    if pods.get(ip).is_some_and(|known| *known != pod) {
                        shared.push(*ip);
                    }
                    pods.insert(*ip, pod);
                }
            }
            for ip in shared {
                pods.remove(&ip);
            }
            let bridges: HashSet<IpAddr> = self
                .bridge_slots
                .lock()
                .values()
                .flat_map(|listing| listing.pods.iter().map(|(ip, _)| *ip))
                .filter(|ip| !pods.contains_key(ip))
                .collect();
            let n = pods.len();
            *self.pods.write() = pods;
            *self.bridges.write() = bridges;
            n
        };
        match errors.is_empty() {
            true => Ok(n),
            false => Err(errors.join("; ")),
        }
    }

    async fn list(
        &self,
        namespace: &str,
        selector: &str,
        token: Option<&str>,
    ) -> Result<Vec<(IpAddr, String)>, String> {
        let encode =
            |s: &str| url::form_urlencoded::byte_serialize(s.as_bytes()).collect::<String>();
        let path = format!(
            "/api/v1/namespaces/{}/pods?labelSelector={}",
            encode(namespace),
            encode(selector)
        );
        self.http
            .call(http::Method::GET, &path, token, None)
            .await
            .map(|list| pods_of(&list))
            .map_err(|e| e.to_string())
    }

    pub fn refused(&self) -> Option<String> {
        self.refused.read().clone()
    }

    pub fn bridges(&self) -> HashSet<IpAddr> {
        self.bridges.read().clone()
    }

    pub fn live(&self) -> HashMap<i64, HashSet<String>> {
        self.slots
            .lock()
            .iter()
            .map(|(id, listing)| {
                (
                    *id,
                    listing.pods.iter().map(|(_, uid)| uid.clone()).collect(),
                )
            })
            .collect()
    }
}

impl PeerDirectory for KubePods {
    fn lookup(&self, ip: IpAddr) -> Option<(i64, String)> {
        self.pods.read().get(&ip).cloned()
    }
}

pub async fn pod_watch_once(gateway: &Gateway, pods: &KubePods) -> Option<Result<usize, String>> {
    let policy = gateway.policy()?;
    let listed = pods.refresh(&policy).await;
    gateway.note_pods_refused(pods.refused());
    gateway.note_live_pods(pods.live());
    gateway.note_bridge_pods(pods.bridges());
    gateway.identify_peers(pods);
    Some(listed)
}

pub async fn run_pod_watch(gateway: Arc<Gateway>, pods: Arc<KubePods>, every: Duration) {
    let mut known = None;
    loop {
        match pod_watch_once(&gateway, &pods).await {
            Some(Ok(n)) if known != Some(n) => {
                info!("gateway: {n} pod addresses known");
                known = Some(n);
            }
            Some(Err(e)) => warn!("gateway: listing pods: {e}"),
            _ => {}
        }
        tokio::time::sleep(every).await;
    }
}

pub async fn run_policy_loop(gateway: Arc<Gateway>, client: PolicyClient, every: Duration) {
    loop {
        match client.fetch().await {
            Ok(raw) => {
                let g = gateway.clone();
                let applied =
                    tokio::task::spawn_blocking(move || g.apply(&raw, Instant::now())).await;
                match applied {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => warn!("gateway: policy not applied: {e}"),
                    Err(e) => warn!("gateway: applying the policy panicked: {e}"),
                }
            }
            Err(e) => warn!("gateway: reading the policy: {e}"),
        }
        if let Some(body) = gateway.ack()
            && let Err(e) = client.ack(body).await
        {
            warn!("gateway: posting the ack: {e}");
        }
        tokio::time::sleep(every).await;
    }
}

pub async fn run_durable_loop(gateway: Arc<Gateway>, every: Duration) {
    loop {
        let g = gateway.clone();
        match tokio::task::spawn_blocking(move || g.reconcile_durable()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => warn!("gateway: the state dir: {e}"),
            Err(e) => warn!("gateway: reconciling the state dir panicked: {e}"),
        }
        tokio::select! {
            _ = gateway.wake_durable.notified() => {}
            _ = tokio::time::sleep(every) => {}
        }
    }
}

pub fn trainer_probe_limit(request_timeout: Duration) -> Duration {
    let connect = std::env::var("COPY_PORT_CONNECT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    Duration::from_secs(connect) + request_timeout
}

pub async fn trainer_newest(copy_url: &str, limit: Duration) -> Option<String> {
    let probe = crate::copy_port_client::probe_newer_checkpoint(0, copy_url);
    match tokio::time::timeout(limit, probe).await {
        Ok(Ok(found)) => found.map(|(prefix, _)| prefix),
        Ok(Err(e)) => {
            warn!("gateway: listing the trainer: {e}");
            None
        }
        Err(_) => {
            warn!("gateway: the trainer was not listed within {limit:?}; polling the proxy");
            None
        }
    }
}

pub async fn run_checkpoint_loop(
    gateway: Arc<Gateway>,
    copy_url: String,
    mut proxy: crate::checkpoint_proxy::CheckpointProxy,
    make_proxy: impl Fn() -> crate::checkpoint_proxy::CheckpointProxy,
    poll_interval: Duration,
    probe_limit: Duration,
) {
    const MAX_CONSECUTIVE_STORAGE_FULL: u32 = 5;
    let mut storage_full_streak: u32 = 0;
    loop {
        let mark = gateway.probe_mark();
        let newest = trainer_newest(&copy_url, probe_limit).await;
        if let Some(newest) = &newest
            && gateway.rewind_to(newest)
        {
            proxy = make_proxy();
        }
        let polled = if newest.as_deref().is_some_and(|n| gateway.mirrors(n)) {
            Ok(false)
        } else {
            proxy.poll_and_download().await
        };
        match polled {
            Ok(downloaded) => {
                storage_full_streak = 0;
                if downloaded {
                    info!("poll: new checkpoint downloaded successfully");
                }
            }
            Err(e) => {
                let storage_full = e
                    .downcast_ref::<io::Error>()
                    .is_some_and(|io_err| io_err.kind() == io::ErrorKind::StorageFull);
                if storage_full {
                    storage_full_streak += 1;
                    warn!(
                        "poll: storage full ({}/{}): {}",
                        storage_full_streak, MAX_CONSECUTIVE_STORAGE_FULL, e
                    );
                    if storage_full_streak >= MAX_CONSECUTIVE_STORAGE_FULL {
                        log::error!(
                            "poll: {} consecutive storage-full polls; restarting to release \
                             mmap'd tmpfs pages and start from a clean slate",
                            storage_full_streak
                        );
                        std::process::exit(1);
                    }
                } else {
                    storage_full_streak = 0;
                    warn!("poll: error: {}", e);
                }
            }
        }
        if let Some(newest) = &newest {
            gateway.note_mirrored_probed(newest, mark);
        }
        tokio::time::sleep(poll_interval).await;
    }
}

#[cfg(test)]
#[path = "checkpoint_gateway_tests.rs"]
mod tests;
