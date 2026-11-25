// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub const SEQ_WIDTH: usize = 18;

const OFFERS_KEPT_PER_PEER: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Follow,
    Hold,
    Canary,
    Rolling,
}

impl Mode {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "follow" => Ok(Self::Follow),
            "hold" => Ok(Self::Hold),
            "canary" => Ok(Self::Canary),
            "rolling" => Ok(Self::Rolling),
            other => Err(format!("unknown mode {other:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Follow => "follow",
            Self::Hold => "hold",
            Self::Canary => "canary",
            Self::Rolling => "rolling",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Held {
    pub prefix: String,
    pub serving_seq: u64,
    pub nfs_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Released {
    pub run: String,
    pub generation: u64,
    pub seq_offset: i64,
    pub from: Option<String>,
}

impl Released {
    pub fn key(&self) -> String {
        format!("{}-g{}", self.run, self.generation)
    }

    pub fn is_runs(&self, real: &str) -> bool {
        parse_prefix(real)
            .is_some_and(|(_, p)| p.split('/').nth(1).is_some_and(|r| same_run(r, &self.run)))
    }
}

pub fn same_run(a: &str, b: &str) -> bool {
    fn base(id: &str) -> &str {
        match id.rsplit_once('-') {
            Some((base, n))
                if !base.is_empty() && !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) =>
            {
                base
            }
            _ => id,
        }
    }
    base(a) == base(b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub pods: usize,
    pub per_minutes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SlotPolicy {
    pub slot_id: i64,
    pub namespace: String,
    pub pod_selector: Value,
    pub released: bool,
    pub admitted: Vec<String>,
    pub canary_pods: Option<usize>,
    pub budget: Option<Budget>,
    pub bridge: Option<SlotBridge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SlotBridge {
    pub namespace: String,
    pub pod_selector: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bridge {
    pub name: String,
    pub client_cn: String,
    pub released: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NfsPolicy {
    pub dir: String,
    pub offered: Option<String>,
    pub offered_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ceiling {
    pub prefixes: Vec<String>,
}

impl Ceiling {
    fn allows(&self, real: &str) -> bool {
        let run = |p: &str| p.split('/').nth(1).map(str::to_string);
        let Some((elapsed, real)) = parse_prefix(real) else {
            return false;
        };
        let Some(name) = run(&real) else {
            return false;
        };
        self.prefixes
            .iter()
            .filter_map(|p| parse_prefix(p))
            .any(|(cap, cap_real)| {
                elapsed <= cap && run(&cap_real).is_some_and(|c| same_run(&name, &c))
            })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    pub version: u64,
    pub binding: Value,
    pub mode: Mode,
    pub upstream: String,
    pub held: Option<Held>,
    pub released: Option<Released>,
    pub slots: Vec<SlotPolicy>,
    pub bridges: Vec<Bridge>,
    pub nfs: Option<NfsPolicy>,
    pub ceiling: Option<Ceiling>,
    pub raw: Value,
}

fn field<'a>(obj: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    obj.get(key).filter(|v| !v.is_null())
}

fn str_field(obj: &Map<String, Value>, key: &str, what: &str) -> Result<String, String> {
    field(obj, key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{what}.{key} must be a string"))
}

fn u64_field(obj: &Map<String, Value>, key: &str, what: &str) -> Result<u64, String> {
    field(obj, key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{what}.{key} must be a non-negative integer"))
}

fn object<'a>(v: &'a Value, what: &str) -> Result<&'a Map<String, Value>, String> {
    v.as_object()
        .ok_or_else(|| format!("{what} must be an object"))
}

fn array<'a>(obj: &'a Map<String, Value>, key: &str) -> Result<&'a [Value], String> {
    match field(obj, key) {
        None => Ok(&[]),
        Some(Value::Array(items)) => Ok(items),
        Some(_) => Err(format!("{key} must be an array")),
    }
}

impl Policy {
    pub fn from_json(raw: &Value) -> Result<Self, String> {
        let obj = object(raw, "policy")?;
        let version = u64_field(obj, "version", "policy")?;
        let mode = Mode::parse(&str_field(obj, "mode", "policy")?)?;
        let upstream = field(obj, "upstream")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let held = match field(obj, "held") {
            None => None,
            Some(v) => {
                let h = object(v, "held")?;
                let prefix = str_field(h, "prefix", "held")?;
                let serving_seq = u64_field(h, "servingSeq", "held")?;
                let (seq, _) = parse_prefix(&prefix).ok_or_else(|| {
                    format!("held.prefix {prefix:?} is not elapsed_samples_<N>/<name>")
                })?;
                if seq != serving_seq {
                    return Err(format!(
                        "held.prefix {prefix:?} does not carry held.servingSeq {serving_seq}"
                    ));
                }
                check_serving_name(&prefix, "held.prefix")?;
                let nfs_path = field(h, "nfsPath")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                Some(Held {
                    prefix: normalize_prefix(&prefix),
                    serving_seq,
                    nfs_path,
                })
            }
        };
        if mode != Mode::Follow && held.is_none() {
            return Err(format!("mode {} needs held", mode.as_str()));
        }
        let released = match field(obj, "released") {
            None => None,
            Some(v) => {
                let r = object(v, "released")?;
                let run = str_field(r, "run", "released")?;
                if run.is_empty() || run.contains('/') {
                    return Err(format!("released.run {run:?} must be one path component"));
                }
                let seq_offset = field(r, "seqOffset")
                    .and_then(Value::as_i64)
                    .ok_or("released.seqOffset must be an integer")?;
                let from = match field(r, "from") {
                    None => None,
                    Some(v) => {
                        let from = v.as_str().map(normalize_prefix).unwrap_or_default();
                        if !is_real_prefix(&from) {
                            return Err(format!(
                                "released.from {v} is not elapsed_samples_<E>/<name>"
                            ));
                        }
                        Some(from)
                    }
                };
                Some(Released {
                    run,
                    generation: u64_field(r, "generation", "released")?,
                    seq_offset,
                    from,
                })
            }
        };
        let mut slots = Vec::new();
        for (i, v) in array(obj, "slots")?.iter().enumerate() {
            let what = format!("slots[{i}]");
            let s = object(v, &what)?;
            let budget = match field(s, "budget") {
                None => None,
                Some(b) => {
                    let b = object(b, &format!("{what}.budget"))?;
                    Some(Budget {
                        pods: u64_field(b, "pods", &what)? as usize,
                        per_minutes: u64_field(b, "perMinutes", &what)?.max(1),
                    })
                }
            };
            slots.push(SlotPolicy {
                slot_id: field(s, "slotId")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| format!("{what}.slotId must be an integer"))?,
                namespace: field(s, "namespace")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                pod_selector: field(s, "podSelector").cloned().unwrap_or(Value::Null),
                released: field(s, "released")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                admitted: array(s, "admitted")?
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect(),
                canary_pods: field(s, "canaryPods")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize),
                budget,
                bridge: match field(s, "bridge").filter(|b| !b.is_null()) {
                    None => None,
                    Some(b) => {
                        let b = object(b, &format!("{what}.bridge"))?;
                        Some(SlotBridge {
                            namespace: field(b, "namespace")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            pod_selector: field(b, "podSelector").cloned().unwrap_or(Value::Null),
                        })
                    }
                },
            });
        }
        let mut bridges = Vec::new();
        for (i, v) in array(obj, "bridges")?.iter().enumerate() {
            let what = format!("bridges[{i}]");
            let b = object(v, &what)?;
            bridges.push(Bridge {
                name: str_field(b, "name", &what)?,
                client_cn: str_field(b, "clientCn", &what)?,
                released: field(b, "released")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
        }
        let nfs = match field(obj, "nfs") {
            None => None,
            Some(v) => {
                let n = object(v, "nfs")?;
                let offered = field(n, "offered").and_then(Value::as_str);
                if let Some(offered) = offered {
                    check_serving_name(offered, "nfs.offered")?;
                }
                Some(NfsPolicy {
                    dir: str_field(n, "dir", "nfs")?,
                    offered: offered.map(normalize_prefix),
                    offered_path: field(n, "offeredPath")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                })
            }
        };
        let ceiling = match field(obj, "ceiling") {
            None => None,
            Some(v) => {
                let c = object(v, "ceiling")?;
                let listed = match field(c, "prefixes") {
                    None => &[][..],
                    Some(Value::Array(listed)) => listed.as_slice(),
                    Some(other) => {
                        return Err(format!("ceiling.prefixes {other} is not a list"));
                    }
                };
                let mut prefixes = Vec::with_capacity(listed.len());
                for p in listed {
                    let prefix = p.as_str().map(normalize_prefix).unwrap_or_default();
                    if !is_real_prefix(&prefix) {
                        return Err(format!(
                            "ceiling.prefixes {p} is not elapsed_samples_<E>/<name>"
                        ));
                    }
                    prefixes.push(prefix);
                }
                Some(Ceiling { prefixes })
            }
        };
        Ok(Self {
            version,
            binding: field(obj, "binding").cloned().unwrap_or(Value::Null),
            mode,
            upstream,
            held,
            released,
            slots,
            bridges,
            nfs,
            ceiling,
            raw: raw.clone(),
        })
    }

    pub fn binding_key(&self) -> Option<String> {
        binding_key(&self.binding)
    }

    fn slot(&self, slot_id: i64) -> Option<&SlotPolicy> {
        self.slots.iter().find(|s| s.slot_id == slot_id)
    }

    pub fn nfs_seq_limit(&self) -> Option<u64> {
        let all_released = !self.slots.is_empty() && self.slots.iter().all(|s| s.released);
        match (self.mode, &self.held) {
            (Mode::Follow, _) => None,
            (Mode::Rolling, _) if all_released => None,
            (_, Some(held)) => Some(held.serving_seq),
            (_, None) => Some(0),
        }
    }
}

pub fn parse_prefix(prefix: &str) -> Option<(u64, String)> {
    let mut parts = prefix.trim_matches('/').split('/');
    let top = parts.next()?;
    let name = parts.next().filter(|n| !n.is_empty())?;
    let n = top.strip_prefix("elapsed_samples_")?;
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((n.parse().ok()?, format!("{top}/{name}")))
}

pub fn is_real_prefix(prefix: &str) -> bool {
    let mut parts = prefix.split('/');
    let (Some(top), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    top.strip_prefix("elapsed_samples_")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        && !matches!(name, "" | "." | "..")
        && !name.contains('\0')
}

fn normalize_prefix(prefix: &str) -> String {
    prefix.trim_matches('/').to_string()
}

pub fn binding_key(binding: &Value) -> Option<String> {
    match binding {
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

fn check_serving_name(prefix: &str, what: &str) -> Result<(), String> {
    let normalized = normalize_prefix(prefix);
    let two_parts = normalized.split('/').count() == 2;
    match parse_prefix(&normalized) {
        Some((seq, _))
            if two_parts
                && normalized.starts_with(&format!("elapsed_samples_{seq:0SEQ_WIDTH$}/")) =>
        {
            Ok(())
        }
        _ => Err(format!(
            "{what} {prefix:?} is not elapsed_samples_<N, {SEQ_WIDTH} digits>/<name>, which pods load"
        )),
    }
}

pub fn prefix_of(name: &str) -> Option<&str> {
    let i = name.find('/')?;
    let j = name[i + 1..].find('/')?;
    Some(&name[..i + 1 + j])
}

pub fn serving_prefix(seq: u64, run: &str, generation: u64) -> String {
    format!("elapsed_samples_{seq:0SEQ_WIDTH$}/{run}-g{generation}")
}

pub fn relabel(real: &str, released: Option<&Released>) -> Option<(String, u64)> {
    let (elapsed, real) = parse_prefix(real)?;
    let Some(r) = released else {
        return Some((real, elapsed));
    };
    let seq = i128::from(elapsed) + i128::from(r.seq_offset);
    let seq = u64::try_from(seq).ok()?;
    Some((serving_prefix(seq, &r.run, r.generation), seq))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Upstream(String),
    Held(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offer {
    pub serving: String,
    pub seq: u64,
    pub source: Source,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Peer {
    Pod { slot_id: i64, uid: String },
    Unknown(String),
}

impl Peer {
    pub fn key(&self) -> String {
        match self {
            Self::Pod { uid, .. } => format!("pod:{uid}"),
            Self::Unknown(addr) => format!("ip:{addr}"),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Inventory {
    pub latest: Option<String>,
    pub held: Option<String>,
    pub pinned: Vec<String>,
    pub followed: Option<String>,
    pub mirrored: Vec<String>,
}

impl Inventory {
    fn serves(&self, source: &Source) -> bool {
        match source {
            Source::Upstream(real) => self.latest.as_deref() == Some(real.as_str()),
            Source::Held(real) => self.pinned.iter().any(|p| p == real),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Downloaded {
    pub prefix: String,
    pub bytes: u64,
    pub rdma_requested: u64,
    pub ok: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PeerRecord {
    pub slot_id: Option<i64>,
    pub uid: Option<String>,
    pub offers: Vec<Offer>,
    pub admitted: Option<(String, u64)>,
    pub downloaded: Option<Downloaded>,
    pub seen: u64,
}

impl PeerRecord {
    fn max_offer(&self) -> Option<&Offer> {
        self.offers.iter().max_by_key(|o| o.seq)
    }

    fn remember(&mut self, offer: &Offer) {
        self.offers.retain(|o| o.serving != offer.serving);
        self.offers.push(offer.clone());
        if self.offers.len() > OFFERS_KEPT_PER_PEER {
            let drop = self.offers.len() - OFFERS_KEPT_PER_PEER;
            self.offers.drain(..drop);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ledger {
    pub peers: BTreeMap<String, PeerRecord>,
    pub names: BTreeMap<String, String>,
    pub live: Option<BTreeMap<i64, BTreeSet<String>>>,
    pub aliases: BTreeMap<String, (String, u64)>,
    pub bridges: BTreeSet<String>,
}

const NAMES_KEPT: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct Decide {
    pub fresh: bool,
    pub may_admit: bool,
    pub now_secs: u64,
}

impl Ledger {
    pub fn real_of(&self, serving: &str) -> Option<&str> {
        self.names.get(serving).map(String::as_str)
    }

    fn key_of(&self, peer: &Peer) -> String {
        let key = peer.key();
        match self.aliases.get(&key) {
            Some((pod, _)) if !self.peers.contains_key(&key) && self.peers.contains_key(pod) => {
                pod.clone()
            }
            _ => key,
        }
    }

    fn key_of_send(&mut self, peer: &Peer, now_secs: u64) -> String {
        let key = self.key_of(peer);
        if let Some((_, used)) = self
            .aliases
            .get_mut(&peer.key())
            .filter(|(pod, _)| *pod == key)
        {
            *used = (*used).max(now_secs);
        }
        key
    }

    fn is_live(&self, slot_id: i64, uid: &str) -> bool {
        self.live
            .as_ref()
            .is_none_or(|live| live.get(&slot_id).is_some_and(|uids| uids.contains(uid)))
    }

    fn admitted_locally(&self, slot_id: i64, key: &str) -> Vec<(&str, u64)> {
        self.peers
            .values()
            .filter(|r| r.slot_id == Some(slot_id))
            .filter_map(|r| match (&r.uid, &r.admitted) {
                (Some(uid), Some((k, at))) if k == key => Some((uid.as_str(), *at)),
                _ => None,
            })
            .filter(|(uid, _)| self.is_live(slot_id, uid))
            .collect()
    }

    fn admitted(&self, slot: &SlotPolicy, key: &str) -> BTreeSet<String> {
        let mut all: BTreeSet<String> = slot.admitted.iter().cloned().collect();
        all.extend(
            self.admitted_locally(slot.slot_id, key)
                .into_iter()
                .map(|(uid, _)| uid.to_string()),
        );
        all.retain(|uid| self.is_live(slot.slot_id, uid));
        all
    }

    pub fn decide(
        &mut self,
        policy: Option<&Policy>,
        peer: &Peer,
        inv: &Inventory,
        how: Decide,
    ) -> Option<Offer> {
        let key = peer.key();
        let record = self.peers.entry(key.clone()).or_default();
        record.seen = record.seen.max(how.now_secs);
        if let Peer::Pod { slot_id, uid } = peer {
            record.slot_id = Some(*slot_id);
            record.uid = Some(uid.clone());
        }
        let offer = match policy {
            None => cold_start(inv),
            Some(p) => self.by_policy(p, peer, inv, how),
        };
        let record = self.peers.get_mut(&key).expect("inserted above");
        let previous = record.max_offer().cloned();
        let offer = match (policy.map(|p| p.mode), offer, previous) {
            (Some(Mode::Hold), offer, _) => offer,
            (_, Some(o), Some(prev)) if prev.seq > o.seq && inv.serves(&prev.source) => Some(prev),
            (_, None, Some(prev)) if inv.serves(&prev.source) => Some(prev),
            (_, offer, _) => offer,
        };
        if let Some(o) = &offer {
            record.remember(o);
            let real = match &o.source {
                Source::Upstream(r) | Source::Held(r) => r.clone(),
            };
            self.names.insert(o.serving.clone(), real);
            if self.names.len() > NAMES_KEPT {
                let first = self.names.keys().next().cloned().expect("non-empty");
                self.names.remove(&first);
            }
        }
        offer
    }

    fn by_policy(
        &mut self,
        p: &Policy,
        peer: &Peer,
        inv: &Inventory,
        how: Decide,
    ) -> Option<Offer> {
        let held = held_offer(p, inv);
        let candidate = candidate_offer(p, inv);
        match p.mode {
            Mode::Hold => held,
            Mode::Follow => candidate.or(held),
            Mode::Canary | Mode::Rolling => {
                let (Some(candidate), Some(released)) = (candidate, &p.released) else {
                    return held;
                };
                let Peer::Pod { slot_id, uid } = peer else {
                    return held;
                };
                let Some(slot) = p.slot(*slot_id) else {
                    return held;
                };
                let rkey = released.key();
                let admitted = self.admitted(slot, &rkey);
                if admitted.contains(uid) {
                    return Some(candidate);
                }
                if !(how.fresh && how.may_admit) {
                    return held;
                }
                let room = match (p.mode, slot.released) {
                    (Mode::Canary, _) | (Mode::Rolling, false) => {
                        p.mode == Mode::Canary
                            && slot.canary_pods.is_some_and(|n| admitted.len() < n)
                    }
                    (Mode::Rolling, true) => match slot.budget {
                        None => true,
                        Some(b) => {
                            let since = how.now_secs.saturating_sub(b.per_minutes * 60);
                            let recent = self
                                .admitted_locally(slot.slot_id, &rkey)
                                .into_iter()
                                .filter(|(_, at)| *at > since)
                                .count();
                            recent < b.pods
                        }
                    },
                    (Mode::Follow | Mode::Hold, _) => false,
                };
                if !room {
                    return held;
                }
                let record = self.peers.get_mut(&peer.key()).expect("inserted by decide");
                record.admitted = Some((rkey, how.now_secs));
                Some(candidate)
            }
        }
    }

    pub fn identify(
        &mut self,
        lookup: impl Fn(std::net::IpAddr) -> Option<(i64, String)>,
        now_secs: u64,
        ttl_secs: u64,
    ) -> bool {
        let named: Vec<(String, i64, String)> = self
            .peers
            .keys()
            .filter_map(|key| {
                let ip = key.strip_prefix("ip:")?.parse().ok()?;
                let (slot_id, uid) = lookup(ip)?;
                Some((key.clone(), slot_id, uid))
            })
            .collect();
        let mut changed = !named.is_empty();
        for (key, slot_id, uid) in named {
            let Some(early) = self.peers.remove(&key) else {
                continue;
            };
            let pod = Peer::Pod {
                slot_id,
                uid: uid.clone(),
            };
            let record = self.peers.entry(pod.key()).or_default();
            record.slot_id = Some(slot_id);
            record.uid = Some(uid);
            let later = std::mem::take(&mut record.offers);
            for offer in early.offers.iter().chain(&later) {
                record.remember(offer);
            }
            if early.downloaded.is_some()
                && (record.downloaded.is_none() || early.seen > record.seen)
            {
                record.downloaded = early.downloaded;
            }
            record.seen = record.seen.max(early.seen);
            self.aliases.insert(key, (pod.key(), now_secs));
        }
        changed |= self.forget_unknown(now_secs, ttl_secs);
        changed
    }

    pub fn forget_unknown(&mut self, now_secs: u64, ttl_secs: u64) -> bool {
        let before = self.peers.len();
        self.peers
            .retain(|_, r| r.uid.is_some() || r.seen.saturating_add(ttl_secs) >= now_secs);
        self.aliases.retain(|_, (pod, used)| {
            self.peers.contains_key(pod) && used.saturating_add(ttl_secs) >= now_secs
        });
        self.peers.len() != before
    }

    pub fn note_download(
        &mut self,
        peer: &Peer,
        prefix: &str,
        bytes: u64,
        rdma_requested: u64,
        ok: bool,
        now_secs: u64,
    ) {
        let key = self.key_of_send(peer, now_secs);
        let record = self.peers.entry(key).or_default();
        record.seen = record.seen.max(now_secs);
        match &mut record.downloaded {
            Some(d) if d.prefix == prefix => {
                d.bytes += bytes;
                d.rdma_requested += rdma_requested;
                d.ok &= ok;
            }
            d => {
                *d = Some(Downloaded {
                    prefix: prefix.to_string(),
                    bytes,
                    rdma_requested,
                    ok,
                })
            }
        }
    }

    pub fn allowed(&mut self, peer: &Peer, serving: &str, now_secs: u64) -> Option<Source> {
        let key = self.key_of_send(peer, now_secs);
        let record = self.peers.get_mut(&key)?;
        let source = record
            .offers
            .iter()
            .find(|o| o.serving == serving)
            .map(|o| o.source.clone())?;
        record.seen = record.seen.max(now_secs);
        Some(source)
    }

    pub fn forget_download(&mut self, peer: &Peer, prefix: &str) {
        let key = self.key_of(peer);
        if let Some(record) = self.peers.get_mut(&key)
            && record
                .downloaded
                .as_ref()
                .is_some_and(|d| d.prefix == prefix && d.ok)
        {
            record.downloaded = None;
        }
    }

    pub fn ack(&self) -> Value {
        let mut offered = Map::new();
        let mut downloaded = Map::new();
        let mut unknown = 0u64;
        let mut bridges = 0u64;
        let mut seq_max = 0u64;
        let mut pods_seq_max = 0u64;
        for (key, record) in &self.peers {
            if let (Some(slot_id), Some(uid)) = (record.slot_id, &record.uid)
                && !self.is_live(slot_id, uid)
            {
                continue;
            }
            if let Some(o) = record.max_offer() {
                seq_max = seq_max.max(o.seq);
            }
            if let Some((seq, _)) = record
                .downloaded
                .as_ref()
                .filter(|d| d.ok)
                .and_then(|d| parse_prefix(&d.prefix))
            {
                pods_seq_max = pods_seq_max.max(seq);
            }
            let Some(uid) = &record.uid else {
                match self.bridges.contains(key) {
                    true => bridges += 1,
                    false => unknown += 1,
                }
                continue;
            };
            if let Some(o) = record.offers.last() {
                offered.insert(uid.clone(), json!(o.serving));
            }
            if let Some(d) = &record.downloaded {
                downloaded.insert(
                    uid.clone(),
                    json!({
                        "prefix": d.prefix,
                        "bytes": d.bytes,
                        "rdmaRequestedBytes": d.rdma_requested,
                        "ok": d.ok,
                    }),
                );
            }
        }
        json!({
            "offered": offered,
            "downloaded": downloaded,
            "unknownPeers": unknown,
            "bridgePeers": bridges,
            "servingSeqMax": seq_max,
            "podsSeqMax": pods_seq_max,
        })
    }

    pub fn to_json(&self) -> Value {
        let source = |s: &Source| match s {
            Source::Upstream(r) => json!({"upstream": r}),
            Source::Held(r) => json!({"held": r}),
        };
        let peers: Map<String, Value> = self
            .peers
            .iter()
            .map(|(k, r)| {
                (
                    k.clone(),
                    json!({
                        "slotId": r.slot_id,
                        "uid": r.uid,
                        "offers": r.offers.iter().map(|o| json!({
                            "serving": o.serving, "seq": o.seq, "source": source(&o.source),
                        })).collect::<Vec<_>>(),
                        "admitted": r.admitted.as_ref().map(|(k, at)| json!({"key": k, "at": at})),
                        "downloaded": r.downloaded.as_ref().map(|d| json!({
                            "prefix": d.prefix, "bytes": d.bytes,
                            "rdmaRequested": d.rdma_requested, "ok": d.ok,
                        })),
                        "seen": r.seen,
                    }),
                )
            })
            .collect();
        json!({"peers": peers, "names": self.names})
    }

    pub fn from_json(v: &Value) -> Self {
        let mut ledger = Self::default();
        if let Some(names) = v.get("names").and_then(Value::as_object) {
            for (k, real) in names {
                if let Some(real) = real.as_str().filter(|r| is_real_prefix(r))
                    && is_real_prefix(k)
                {
                    ledger.names.insert(k.clone(), real.to_string());
                }
            }
        }
        let Some(peers) = v.get("peers").and_then(Value::as_object) else {
            return ledger;
        };
        for (k, r) in peers {
            let offers = r
                .get("offers")
                .and_then(Value::as_array)
                .map(|offers| {
                    offers
                        .iter()
                        .filter_map(|o| {
                            let src = o.get("source")?;
                            let source =
                                if let Some(r) = src.get("upstream").and_then(Value::as_str) {
                                    Source::Upstream(r.to_string())
                                } else {
                                    Source::Held(src.get("held")?.as_str()?.to_string())
                                };
                            let serving = o.get("serving")?.as_str()?;
                            let (Source::Upstream(real) | Source::Held(real)) = &source;
                            if !(is_real_prefix(serving) && is_real_prefix(real)) {
                                return None;
                            }
                            Some(Offer {
                                serving: serving.to_string(),
                                seq: o.get("seq")?.as_u64()?,
                                source,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            ledger.peers.insert(
                k.clone(),
                PeerRecord {
                    slot_id: r.get("slotId").and_then(Value::as_i64),
                    uid: r.get("uid").and_then(Value::as_str).map(str::to_string),
                    offers,
                    admitted: r.get("admitted").and_then(|a| {
                        Some((a.get("key")?.as_str()?.to_string(), a.get("at")?.as_u64()?))
                    }),
                    downloaded: r.get("downloaded").and_then(|d| {
                        Some(Downloaded {
                            prefix: d
                                .get("prefix")?
                                .as_str()
                                .filter(|p| is_real_prefix(p))?
                                .to_string(),
                            bytes: d.get("bytes")?.as_u64()?,
                            rdma_requested: d.get("rdmaRequested")?.as_u64()?,
                            ok: d.get("ok")?.as_bool()?,
                        })
                    }),
                    seen: r.get("seen").and_then(Value::as_u64).unwrap_or(0),
                },
            );
        }
        ledger
    }
}

fn cold_start(inv: &Inventory) -> Option<Offer> {
    let real = inv.pinned.last()?;
    let (seq, serving) = parse_prefix(real)?;
    Some(Offer {
        serving,
        seq,
        source: Source::Held(real.clone()),
    })
}

fn held_offer(p: &Policy, inv: &Inventory) -> Option<Offer> {
    let held = p.held.as_ref()?;
    Some(Offer {
        serving: held.prefix.clone(),
        seq: held.serving_seq,
        source: Source::Held(inv.held.clone()?),
    })
}

pub(crate) fn candidate_offer(p: &Policy, inv: &Inventory) -> Option<Offer> {
    let latest = inv.latest.as_ref()?;
    let Some(ceiling) = p.ceiling.as_ref().filter(|c| !c.allows(latest)) else {
        return offer_of(p, inv, latest);
    };
    if !matches!(p.mode, Mode::Canary | Mode::Rolling) {
        return None;
    }
    let elapsed = |real: &String| parse_prefix(real).map(|(n, _)| n);
    let mut allowed: Vec<&String> = inv
        .mirrored
        .iter()
        .filter(|real| ceiling.allows(real) && elapsed(real) < elapsed(latest))
        .collect();
    allowed.sort_by_key(|real| std::cmp::Reverse(elapsed(real)));
    allowed.into_iter().find_map(|real| offer_of(p, inv, real))
}

fn offer_of(p: &Policy, inv: &Inventory, latest: &str) -> Option<Offer> {
    let holds = p.mode != Mode::Follow || p.held.is_some();
    if inv.held.as_deref() == Some(latest) && holds {
        return None;
    }
    if let Some(released) = &p.released {
        if let Some(from) = &released.from
            && inv.followed.as_ref() != Some(from)
        {
            return None;
        }
        let pinned_to_run = p.mode != Mode::Follow || released.from.is_some();
        if pinned_to_run && !released.is_runs(latest) {
            return None;
        }
    }
    let (serving, seq) = relabel(latest, p.released.as_ref())?;
    if holds
        && let Some(held) = &p.held
        && seq <= held.serving_seq
    {
        return None;
    }
    Some(Offer {
        serving,
        seq,
        source: Source::Upstream(latest.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn policy(v: Value) -> Policy {
        Policy::from_json(&v).unwrap()
    }

    #[test]
    fn relabel_zero_pads_and_applies_the_offset() {
        let r = Released {
            run: "run-9".into(),
            generation: 2,
            seq_offset: 1_000,
            from: None,
        };
        assert_eq!(
            relabel("elapsed_samples_000000000000000500/run-0", Some(&r)),
            Some((
                "elapsed_samples_000000000000001500/run-9-g2".to_string(),
                1_500
            ))
        );
        assert_eq!(
            relabel("elapsed_samples_000000000000000500/run-0/", None),
            Some(("elapsed_samples_000000000000000500/run-0".to_string(), 500))
        );
        let below_zero = Released {
            seq_offset: -501,
            ..r
        };
        assert_eq!(
            relabel("elapsed_samples_500/run-0", Some(&below_zero)),
            None
        );
    }

    #[test]
    fn a_policy_that_does_not_parse_is_refused_with_why() {
        for (v, why) in [
            (json!({"version": 1, "mode": "hold"}), "needs held"),
            (json!({"version": 1, "mode": "fast"}), "unknown mode"),
            (
                json!({"version": 1, "mode": "hold", "held": {"prefix": "elapsed_samples_5/r-g1", "servingSeq": 6}}),
                "does not carry",
            ),
            (
                json!({"version": 1, "mode": "follow", "released": {"run": "a/b", "generation": 1, "seqOffset": 0}}),
                "one path component",
            ),
            (
                json!({"version": 1, "mode": "follow", "released": {"run": "a", "generation": 1, "seqOffset": 0, "from": "run-0"}}),
                "released.from",
            ),
            (
                json!({"version": 1, "mode": "follow", "released": {"run": "a", "generation": 1, "seqOffset": 0, "from": "elapsed_samples_5/../x"}}),
                "released.from",
            ),
        ] {
            let err = Policy::from_json(&v).unwrap_err();
            assert!(err.contains(why), "{err}");
        }
    }

    #[test]
    fn a_serving_name_pods_would_skip_is_refused() {
        let held = |prefix: &str, seq: u64| json!({"version": 1, "mode": "hold", "held": {"prefix": prefix, "servingSeq": seq}});
        let padded = serving_prefix(1200, "inc", 3);
        for v in [
            held("elapsed_samples_1200/inc-g3", 1200),
            held(&format!("{padded}/extra"), 1200),
            {
                let mut v = held(&padded, 1200);
                v["nfs"] = json!({"dir": "/nfs/serving", "offered": "elapsed_samples_1200/inc-g3"});
                v
            },
        ] {
            let err = Policy::from_json(&v).expect_err(&format!("{v} parsed"));
            assert!(err.contains("18 digits"), "{err}");
        }
        assert!(Policy::from_json(&held(&padded, 1200)).is_ok());
    }

    #[test]
    fn the_ledger_survives_its_json() {
        let mut ledger = Ledger::default();
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let inv = Inventory {
            latest: Some("elapsed_samples_7/run-0".into()),
            ..Default::default()
        };
        let peer = Peer::Pod {
            slot_id: 30,
            uid: "uid-1".into(),
        };
        let how = Decide {
            fresh: true,
            may_admit: true,
            now_secs: 10,
        };
        ledger.decide(Some(&p), &peer, &inv, how).unwrap();
        assert_eq!(Ledger::from_json(&ledger.to_json()), ledger);
    }

    fn real(elapsed: u64) -> String {
        format!("elapsed_samples_{elapsed:018}/inc")
    }

    fn follow_holding(from: Option<String>) -> Policy {
        policy(json!({
            "version": 5,
            "mode": "follow",
            "held": {"prefix": serving_prefix(2000, "inc", 3), "servingSeq": 2000},
            "released": {"run": "inc", "generation": 3, "seqOffset": 1000, "from": from},
        }))
    }

    fn offered(p: &Policy, inv: &Inventory, uid: &str) -> Option<String> {
        let peer = Peer::Pod {
            slot_id: 30,
            uid: uid.into(),
        };
        let how = Decide {
            fresh: true,
            may_admit: true,
            now_secs: 10,
        };
        Ledger::default()
            .decide(Some(p), &peer, inv, how)
            .map(|o| o.serving)
    }

    #[test]
    fn a_follow_that_holds_offers_nothing_at_or_below_its_hold() {
        let p = follow_holding(None);
        let held = serving_prefix(2000, "inc", 3);
        let inv = |latest: u64| Inventory {
            latest: Some(real(latest)),
            held: Some(real(1000)),
            pinned: vec![real(1000)],
            ..Default::default()
        };
        assert_eq!(offered(&p, &inv(900), "uid-1"), Some(held.clone()));
        assert_eq!(offered(&p, &inv(1000), "uid-1"), Some(held));
        assert_eq!(
            offered(&p, &inv(1100), "uid-1"),
            Some(serving_prefix(2100, "inc", 3))
        );
    }

    #[test]
    fn a_follow_from_a_restored_checkpoint_offers_none_of_the_trainers_before_it() {
        let p = follow_holding(Some(real(1100)));
        let held = serving_prefix(2000, "inc", 3);
        let inv = |latest: u64, followed: Option<String>| Inventory {
            latest: Some(real(latest)),
            held: Some(real(1000)),
            pinned: vec![real(1000)],
            followed,
            mirrored: Vec::new(),
        };
        assert_eq!(offered(&p, &inv(1300, None), "fresh"), Some(held.clone()));
        assert_eq!(offered(&p, &inv(1100, None), "fresh"), Some(held.clone()));
        let other = Some(real(1050));
        assert_eq!(offered(&p, &inv(1300, other), "fresh"), Some(held));
        assert_eq!(
            offered(&p, &inv(1100, Some(real(1100))), "fresh"),
            Some(serving_prefix(2100, "inc", 3))
        );
        assert_eq!(
            offered(&p, &inv(1200, Some(real(1100))), "fresh"),
            Some(serving_prefix(2200, "inc", 3))
        );
    }

    #[test]
    fn a_ceiling_offers_nothing_the_gate_holds() {
        let inv = |latest: &str| Inventory {
            latest: Some(latest.to_string()),
            ..Default::default()
        };
        let follow = |ceiling: Value| {
            policy(json!({"version": 1, "mode": "follow",
                          "released": {"run": "inc", "generation": 1, "seqOffset": 0},
                          "ceiling": ceiling}))
        };
        let none = follow(json!({"prefixes": []}));
        assert_eq!(offered(&none, &inv(&real(1000)), "z"), None);
        let capped = follow(json!({"prefixes": ["elapsed_samples_1000/inc"]}));
        assert_eq!(
            offered(&capped, &inv(&real(1000)), "a"),
            Some(serving_prefix(1000, "inc", 1))
        );
        assert_eq!(offered(&capped, &inv(&real(1100)), "b"), None);
        assert_eq!(
            offered(
                &capped,
                &inv("elapsed_samples_000000000000000900/cand"),
                "c"
            ),
            None,
            "another run's checkpoint is not one the gate allowed"
        );
        let plain = follow(Value::Null);
        assert_eq!(
            offered(&plain, &inv(&real(1100)), "d"),
            Some(serving_prefix(1100, "inc", 1))
        );

        let mut canary = follow_holding(None).raw;
        canary["mode"] = json!("canary");
        canary["slots"] = json!([{"slotId": 30, "namespace": "ns", "podSelector": "app=x",
                                  "canaryPods": 2}]);
        canary["ceiling"] = json!({"prefixes": [real(1100)]});
        let canary = policy(canary);
        let inv = |latest: u64| Inventory {
            latest: Some(real(latest)),
            held: Some(real(1000)),
            pinned: vec![real(1000)],
            ..Default::default()
        };
        let held = serving_prefix(2000, "inc", 3);
        assert_eq!(
            offered(&canary, &inv(1100), "e"),
            Some(serving_prefix(2100, "inc", 3))
        );
        assert_eq!(offered(&canary, &inv(1200), "f"), Some(held));
        for bad in [
            json!("elapsed_samples_x"),
            json!({"prefixes": "elapsed_samples_1000/inc"}),
            json!({"prefixes": ["elapsed_samples_x"]}),
        ] {
            let e = Policy::from_json(&json!({"version": 1, "mode": "follow", "ceiling": bad}))
                .unwrap_err();
            assert!(e.contains("ceiling"), "{e}");
        }
    }

    #[test]
    fn a_ceiling_caps_each_run_by_the_gates_newest_of_that_run() {
        let inv = |latest: &str| Inventory {
            latest: Some(latest.to_string()),
            ..Default::default()
        };
        let p = policy(json!({"version": 1, "mode": "follow",
                              "ceiling": {"prefixes": ["elapsed_samples_000000000000000900/cand-3",
                                                       "elapsed_samples_1200/inc-0"]}}));
        assert_eq!(
            offered(&p, &inv("elapsed_samples_000000000000001150/inc-1"), "a"),
            Some("elapsed_samples_000000000000001150/inc-1".into()),
            "the restored incumbent, restarted, is capped by its own run's 1200"
        );
        assert_eq!(
            offered(&p, &inv("elapsed_samples_000000000000001250/inc-1"), "b"),
            None
        );
        assert_eq!(
            offered(&p, &inv("elapsed_samples_000000000000000900/cand-3"), "c"),
            Some("elapsed_samples_000000000000000900/cand-3".into()),
            "the candidate is capped by its own 900, not the incumbent's higher 1200"
        );
        assert_eq!(
            offered(&p, &inv("elapsed_samples_000000000000000950/cand-3"), "d"),
            None
        );
        assert_eq!(
            offered(&p, &inv("elapsed_samples_000000000000000100/other"), "e"),
            None
        );
    }

    #[test]
    fn a_canary_offers_the_newest_allowed_candidate_while_the_trainer_saves_past_the_ceiling() {
        let cand = |elapsed: u64| format!("elapsed_samples_{elapsed:018}/cand");
        let canary = |cap: &str| {
            policy(json!({
                "version": 3,
                "mode": "canary",
                "held": {"prefix": serving_prefix(2000, "inc", 3), "servingSeq": 2000},
                "released": {"run": "cand", "generation": 4, "seqOffset": 1000},
                "slots": [{"slotId": 30, "namespace": "ns", "podSelector": "app=x",
                           "canaryPods": 2, "admitted": ["uid-a"]}],
                "ceiling": {"prefixes": [cap]},
            }))
        };
        let inv = Inventory {
            latest: Some(cand(1300)),
            held: Some(real(1000)),
            pinned: vec![real(1000)],
            mirrored: vec![
                cand(1300),
                cand(1100),
                "elapsed_samples_1150/cand".into(),
                real(1250),
                cand(900),
                cand(1200),
            ],
            ..Default::default()
        };
        let held = serving_prefix(2000, "inc", 3);
        let ask = |ledger: &mut Ledger, p: &Policy, uid: &str| {
            let peer = Peer::Pod {
                slot_id: 30,
                uid: uid.into(),
            };
            ledger
                .decide(Some(p), &peer, &inv, at(10))
                .map(|o| o.serving)
        };

        let p = canary(&cand(1200));
        let mut ledger = Ledger::default();
        let allowed = Some(serving_prefix(2200, "cand", 4));
        assert_eq!(ask(&mut ledger, &p, "uid-a"), allowed, "the admitted pod");
        assert_eq!(ask(&mut ledger, &p, "uid-b"), allowed, "a pod with room");
        assert_eq!(ask(&mut ledger, &p, "uid-c"), Some(held.clone()), "no room");
        assert_eq!(ask(&mut ledger, &p, "uid-b"), allowed, "admitted on it");

        let p = canary("elapsed_samples_1150/cand");
        assert_eq!(
            ask(&mut Ledger::default(), &p, "uid-a"),
            Some(serving_prefix(2150, "cand", 4))
        );
        for cap in [cand(900), real(1250), cand(1000)] {
            let p = canary(&cap);
            assert_eq!(
                ask(&mut Ledger::default(), &p, "uid-a"),
                Some(held.clone()),
                "ceiling {cap}"
            );
        }
        let mut rolling = canary(&cand(1200)).raw;
        rolling["mode"] = json!("rolling");
        rolling["slots"][0]["released"] = json!(true);
        assert_eq!(
            ask(&mut Ledger::default(), &policy(rolling), "uid-z"),
            allowed
        );
    }

    fn at(now_secs: u64) -> Decide {
        Decide {
            fresh: true,
            may_admit: true,
            now_secs,
        }
    }

    #[test]
    fn a_newest_checkpoint_of_another_run_is_never_offered_as_the_released_one() {
        let cand = |elapsed: u64| format!("elapsed_samples_{elapsed:018}/cand");
        let held = serving_prefix(2000, "inc", 3);
        let inv = |latest: String, followed: Option<String>| Inventory {
            latest: Some(latest),
            held: Some(real(1000)),
            pinned: vec![real(1000)],
            followed,
            mirrored: Vec::new(),
        };
        let restored = follow_holding(Some(real(1100)));
        assert_eq!(
            offered(&restored, &inv(cand(1300), Some(real(1100))), "fresh"),
            Some(held.clone()),
            "the rolled-back candidate's checkpoint was offered as the restored run's"
        );
        assert_eq!(
            offered(&restored, &inv(real(1300), Some(real(1100))), "fresh"),
            Some(serving_prefix(2300, "inc", 3))
        );

        let canary = policy(json!({
            "version": 6,
            "mode": "canary",
            "held": {"prefix": held, "servingSeq": 2000},
            "released": {"run": "cand", "generation": 4, "seqOffset": 1000},
            "slots": [{"slotId": 30, "canaryPods": 2}],
        }));
        assert_eq!(
            offered(&canary, &inv(real(1300), None), "uid-1"),
            Some(held.clone())
        );
        assert_eq!(
            offered(&canary, &inv(cand(1300), None), "uid-1"),
            Some(serving_prefix(2300, "cand", 4))
        );

        let finished = follow_holding(None);
        assert_eq!(
            offered(&finished, &inv(cand(1300), None), "fresh"),
            Some(serving_prefix(2300, "inc", 3))
        );
    }

    #[test]
    fn a_ledger_keeps_what_each_pod_downloaded_through_its_json() {
        let mut ledger = Ledger::default();
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let inv = Inventory {
            latest: Some(real(7)),
            ..Default::default()
        };
        let peer = Peer::Pod {
            slot_id: 30,
            uid: "uid-1".into(),
        };
        ledger.decide(Some(&p), &peer, &inv, at(10)).unwrap();
        ledger.note_download(&peer, &real(7), 96, 0, true, 10);
        let back = Ledger::from_json(&ledger.to_json());
        assert_eq!(back, ledger);
        assert_eq!(
            back.ack()["downloaded"]["uid-1"],
            json!({"prefix": real(7), "bytes": 96, "rdmaRequestedBytes": 0, "ok": true})
        );
    }

    #[test]
    fn an_unknown_peer_is_named_once_listed_and_forgotten_once_gone() {
        let mut ledger = Ledger::default();
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let inv = Inventory {
            latest: Some(real(7)),
            ..Default::default()
        };
        let early = Peer::Unknown("10.0.0.7".into());
        let other = Peer::Unknown("10.0.0.8".into());
        ledger.decide(Some(&p), &early, &inv, at(100)).unwrap();
        ledger.note_download(&early, &real(7), 96, 0, true, 100);
        ledger.decide(Some(&p), &other, &inv, at(100)).unwrap();
        assert_eq!(ledger.ack()["unknownPeers"], json!(2));

        let directory = |ip: std::net::IpAddr| {
            (ip.to_string() == "10.0.0.7").then(|| (30, "uid-7".to_string()))
        };
        assert!(ledger.identify(directory, 115, 300));
        let ack = ledger.ack();
        assert_eq!(ack["unknownPeers"], json!(1), "{ack}");
        assert_eq!(ack["offered"]["uid-7"], json!(real(7)), "{ack}");
        assert_eq!(ack["downloaded"]["uid-7"]["ok"], json!(true), "{ack}");

        ledger.decide(Some(&p), &other, &inv, at(380)).unwrap();
        assert!(!ledger.identify(directory, 600, 300));
        assert_eq!(ledger.ack()["unknownPeers"], json!(1));
        assert!(ledger.identify(directory, 681, 300));
        assert_eq!(ledger.ack()["unknownPeers"], json!(0));
    }

    #[test]
    fn naming_a_peer_keeps_the_download_it_made_last() {
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let pod = Peer::Pod {
            slot_id: 30,
            uid: "uid-7".into(),
        };
        let ip = Peer::Unknown("10.0.0.7".into());
        let directory =
            |a: std::net::IpAddr| (a.to_string() == "10.0.0.7").then(|| (30, "uid-7".to_string()));
        let ask = |ledger: &mut Ledger, peer: &Peer, step: u64, now: u64| {
            let inv = Inventory {
                latest: Some(real(step)),
                ..Default::default()
            };
            ledger.decide(Some(&p), peer, &inv, at(now)).unwrap();
            ledger.note_download(peer, &real(step), 96, 0, true, now);
        };
        for (pod_at, ip_at) in [(100, 200), (200, 100)] {
            let mut ledger = Ledger::default();
            let (pod_step, ip_step) = if pod_at < ip_at { (7, 9) } else { (9, 7) };
            ask(&mut ledger, &pod, pod_step, pod_at);
            ask(&mut ledger, &ip, ip_step, ip_at);
            assert!(ledger.identify(directory, 215, 300));
            let ack = ledger.ack();
            assert_eq!(
                ack["downloaded"]["uid-7"]["prefix"],
                json!(real(9)),
                "{ack}"
            );
            assert_eq!(ack["unknownPeers"], json!(0), "{ack}");
        }
    }

    #[test]
    fn the_ack_reports_the_highest_prefix_pods_downloaded() {
        let mut ledger = Ledger::default();
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let pod = |uid: &str| Peer::Pod {
            slot_id: 30,
            uid: uid.into(),
        };
        ledger.note_download(
            &pod("uid-1"),
            &serving_prefix(9201, "w-5", 2),
            10,
            0,
            true,
            10,
        );
        ledger.note_download(
            &pod("uid-2"),
            &serving_prefix(9900, "w-5", 2),
            10,
            0,
            false,
            10,
        );
        let inv = Inventory {
            latest: Some(real(6600)),
            ..Default::default()
        };
        ledger
            .decide(Some(&p), &pod("uid-1"), &inv, at(10))
            .unwrap();
        let ack = ledger.ack();
        assert_eq!(ack["servingSeqMax"], json!(6600), "{ack}");
        assert_eq!(ack["podsSeqMax"], json!(9201), "{ack}");
    }

    #[test]
    fn a_restarted_trainer_is_still_the_released_run() {
        let ckpt = |elapsed: u64, run: &str| format!("elapsed_samples_{elapsed:018}/{run}");
        let held = serving_prefix(2000, "inc-0", 3);
        let inv = |latest: String, followed: Option<String>| Inventory {
            latest: Some(latest),
            held: Some(ckpt(1000, "inc-0")),
            pinned: vec![ckpt(1000, "inc-0")],
            followed,
            mirrored: Vec::new(),
        };
        let restored = policy(json!({
            "version": 5,
            "mode": "follow",
            "held": {"prefix": held, "servingSeq": 2000},
            "released": {"run": "inc-0", "generation": 3, "seqOffset": 1000,
                         "from": ckpt(1100, "inc-0")},
        }));
        let followed = Some(ckpt(1100, "inc-0"));
        assert_eq!(
            offered(
                &restored,
                &inv(ckpt(1200, "inc-1"), followed.clone()),
                "fresh"
            ),
            Some(serving_prefix(2200, "inc-0", 3)),
            "the restored trainer's restart left serving on the restore point"
        );
        assert_eq!(
            offered(&restored, &inv(ckpt(1300, "cand-1"), followed), "fresh"),
            Some(held.clone()),
            "the rolled-back candidate's checkpoint was offered as the restored run's"
        );

        for mode in ["canary", "rolling"] {
            let p = policy(json!({
                "version": 6,
                "mode": mode,
                "held": {"prefix": held, "servingSeq": 2000},
                "released": {"run": "cand-0", "generation": 4, "seqOffset": 1000},
                "slots": [{"slotId": 30, "canaryPods": 2, "released": mode == "rolling"}],
            }));
            assert_eq!(
                offered(&p, &inv(ckpt(1300, "cand-1"), None), "uid-1"),
                Some(serving_prefix(2300, "cand-0", 4)),
                "{mode}: the candidate's restart left its pods on the held checkpoint"
            );
            assert_eq!(
                offered(&p, &inv(ckpt(1300, "inc-1"), None), "uid-1"),
                Some(held.clone()),
                "{mode}: another run's checkpoint was offered as the candidate"
            );
        }
    }

    #[test]
    fn a_send_in_flight_when_its_peer_is_named_reports_to_the_pod() {
        let t = 1_800_000_000;
        let mut ledger = Ledger::default();
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let inv = Inventory {
            latest: Some(real(7)),
            ..Default::default()
        };
        let early = Peer::Unknown("10.0.0.7".into());
        let offer = ledger.decide(Some(&p), &early, &inv, at(t)).unwrap();
        assert!(ledger.allowed(&early, &offer.serving, t + 1).is_some());
        let directory = |ip: std::net::IpAddr| {
            (ip.to_string() == "10.0.0.7").then(|| (30, "uid-7".to_string()))
        };
        assert!(ledger.identify(directory, t + 5, 300));

        ledger.note_download(&early, &offer.serving, 96, 0, true, t + 8);
        ledger.forget_unknown(t + 10, 300);
        let ack = ledger.ack();
        assert_eq!(
            ack["downloaded"]["uid-7"],
            json!({"prefix": offer.serving, "bytes": 96, "rdmaRequestedBytes": 0, "ok": true}),
            "the Send in flight when the pod was named was not counted as its download: {ack}"
        );
        assert_eq!(ack["unknownPeers"], json!(0), "{ack}");
        assert_eq!(
            ledger.allowed(&early, &offer.serving, t + 11),
            Some(Source::Upstream(real(7))),
            "a Send of the same load was refused once the pod was named"
        );
        ledger.note_download(&early, &offer.serving, 0, 0, false, t + 12);
        ledger.forget_unknown(t + 20, 300);
        let ack = ledger.ack();
        assert_eq!(
            ack["downloaded"]["uid-7"]["ok"],
            json!(false),
            "a failed Send in flight was dropped: {ack}"
        );
    }

    #[test]
    fn an_unknown_peer_still_loading_keeps_its_offers() {
        let t = 1_800_000_000;
        let mut ledger = Ledger::default();
        let p = policy(json!({"version": 1, "mode": "follow"}));
        let inv = Inventory {
            latest: Some(real(7)),
            ..Default::default()
        };
        let peer = Peer::Unknown("10.0.0.9".into());
        let offer = ledger.decide(Some(&p), &peer, &inv, at(t)).unwrap();
        let nobody = |_: std::net::IpAddr| None::<(i64, String)>;
        for sent in [200, 400, 600] {
            assert!(
                ledger.allowed(&peer, &offer.serving, t + sent).is_some(),
                "the Send {sent} s after its List was refused"
            );
            ledger.note_download(&peer, &offer.serving, 96, 0, true, t + sent + 50);
            assert!(
                !ledger.identify(nobody, t + sent + 100, 300),
                "a peer still loading was forgotten {} s after its List",
                sent + 100
            );
        }
        assert_eq!(ledger.ack()["unknownPeers"], json!(1));
        assert!(ledger.identify(nobody, t + 1000, 300));
        assert_eq!(ledger.allowed(&peer, &offer.serving, t + 1001), None);
        assert_eq!(ledger.ack()["unknownPeers"], json!(0));
    }
}
