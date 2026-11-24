// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use super::*;
use crate::{
    checkpoint_policy::serving_prefix,
    proto_parser::{bytes, decode_names, encode_names, proto, repeated_ints},
};
use clap::Parser;
use std::net::Ipv4Addr;

const RUN: &str = "run-0";
const CAND: &str = "cand";
const INC: &str = "inc";

fn real(elapsed: u64) -> String {
    real_of(elapsed, RUN)
}

fn real_of(elapsed: u64, run: &str) -> String {
    format!("elapsed_samples_{elapsed:018}/{run}")
}

fn held_policy(version: u64, seq: u64, generation: u64, nfs_path: &str) -> Value {
    json!({
        "version": version,
        "binding": 7,
        "mode": "hold",
        "upstream": "trainer.recsys.svc:9988",
        "held": {"prefix": serving_prefix(seq, "inc", generation), "servingSeq": seq, "nfsPath": nfs_path},
        "released": null,
        "slots": [{"slotId": 30, "namespace": "inference", "podSelector": "app=replies", "released": false, "admitted": []}],
        "bridges": [],
    })
}

fn files(elapsed: u64) -> Vec<(&'static str, Vec<u8>)> {
    let fill = |len: usize, salt: u64| {
        (0..len)
            .map(|i| ((i as u64 * 31 + salt) % 251) as u8)
            .collect()
    };
    vec![
        (
            "checksums.0.json",
            format!("{{\"global_checksums\": {{\"dense\": {elapsed}}}}}").into_bytes(),
        ),
        ("dense_params/w", fill(96, elapsed)),
        ("emb_table/c/0/0", fill(200, elapsed + 1)),
        ("emb_table/c/1/0", fill(200, elapsed + 2)),
    ]
}

fn write_checkpoint(root: &Path, elapsed: u64) {
    write_checkpoint_of(root, elapsed, RUN);
}

fn write_checkpoint_of(root: &Path, elapsed: u64, run: &str) {
    for (name, data) in files(elapsed) {
        let path = root.join(real_of(elapsed, run)).join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, data).unwrap();
    }
}

#[derive(Clone)]
struct FakeCopy {
    root: PathBuf,
    fail_sends: bool,
}

impl Service<Inner> for FakeCopy {
    type Response = http::Response<body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Infallible>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Inner) -> Self::Future {
        let root = self.root.clone();
        let fail_sends = self.fail_sends;
        Box::pin(async move {
            let path = request.uri().path().to_string();
            let mut buf = get_bytes_mut();
            if path == LIST {
                let mut listed = Vec::new();
                for (prefix, files) in scan_prefixes(&root) {
                    listed.extend(
                        files
                            .into_iter()
                            .map(|(f, s)| (format!("{prefix}/{f}").into_bytes(), s)),
                    );
                }
                let (names, sizes): (Vec<_>, Vec<_>) = listed.into_iter().unzip();
                let (all, p, s) = encode_names(names);
                buf.put_string(1, &all);
                buf.put_repeated_ints([2, 3, 4], [&p, &s, &sizes]);
            } else {
                let (mut names, mut p, mut s) =
                    (Vec::new(), Vec::<usize>::new(), Vec::<usize>::new());
                let (mut offsets, mut sizes) = (Vec::<usize>::new(), Vec::<usize>::new());
                let mut endpoints = Vec::new();
                parse(
                    proto![
                        (1, bytes(&mut names)),
                        (2, repeated_ints(&mut p)),
                        (3, repeated_ints(&mut s)),
                        (4, repeated_ints(&mut offsets)),
                        (5, repeated_ints(&mut sizes)),
                        (6, crate::proto_parser::repeated_bytes(&mut endpoints)),
                    ],
                    request.into_body(),
                    BodyKind::Request,
                )
                .await
                .unwrap();
                let mut data = Vec::new();
                for ((name, offset), size) in decode_names(names, p, s)
                    .unwrap()
                    .into_iter()
                    .zip(offsets)
                    .zip(sizes)
                {
                    let name = String::from_utf8(name).unwrap();
                    match fs::read(root.join(&name)) {
                        Ok(file) => data.extend_from_slice(&file[offset..offset + size]),
                        Err(_) => return Ok(make_response(Err(Status::not_found(name)))),
                    }
                }
                if endpoints.is_empty() {
                    buf.put_string(1, &data);
                }
                if fail_sends {
                    let mut trailers = http::HeaderMap::new();
                    Status::internal("the transfer broke")
                        .add_header(&mut trailers)
                        .unwrap();
                    let data = http_body_util::BodyExt::collect(freeze(buf))
                        .await
                        .unwrap()
                        .to_bytes();
                    let frames = [
                        Ok::<_, Infallible>(http_body::Frame::data(data)),
                        Ok(http_body::Frame::trailers(trailers)),
                    ];
                    return Ok(make_response(Ok(body::Body::new(
                        http_body_util::StreamBody::new(futures::stream::iter(frames)),
                    ))));
                }
            }
            Ok(make_response(Ok(add_ok_trailer(freeze(buf)))))
        })
    }
}

fn ip(n: u8) -> IpAddr {
    IpAddr::V4(Ipv4Addr::new(10, 0, 0, n))
}

struct Fixture {
    _dirs: Vec<tempfile::TempDir>,
    root: PathBuf,
    state: PathBuf,
    nfs: PathBuf,
    peers: Arc<StaticPeers>,
}

impl Fixture {
    fn new() -> Self {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let mut pods = HashMap::new();
        for n in 1..=5 {
            pods.insert(ip(n), (30, format!("uid-{n}")));
        }
        pods.insert(ip(16), (16, "uid-16".to_string()));
        Self {
            root: dirs[0].path().to_path_buf(),
            state: dirs[1].path().to_path_buf(),
            nfs: dirs[2].path().to_path_buf(),
            _dirs: dirs,
            peers: Arc::new(StaticPeers(pods)),
        }
    }

    fn config(&self, root: &Path, server_index: usize) -> GatewayConfig {
        GatewayConfig {
            root: root.to_path_buf(),
            state_dir: Some(self.state.clone()),
            server_index,
            policy_fresh: Duration::from_secs(120),
            name: format!("prod-replies-ranker-ckpt-{server_index}"),
            binding: None,
            unknown_peer_ttl: Duration::from_secs(300),
        }
    }

    fn open(&self) -> Arc<Gateway> {
        Arc::new(Gateway::open(self.config(&self.root, 0)))
    }

    fn service(&self, gateway: &Arc<Gateway>, root: &Path) -> GatewayService<FakeCopy> {
        GatewayService::new(
            FakeCopy {
                root: root.to_path_buf(),
                fail_sends: false,
            },
            gateway.clone(),
            self.peers.clone(),
        )
    }

    fn nfs_leaf(&self, elapsed: u64) -> String {
        self.nfs_leaf_of(elapsed, RUN)
    }

    fn nfs_leaf_of(&self, elapsed: u64, run: &str) -> String {
        let leaf = self
            .nfs
            .join(format!("ckpt/elapsed_samples_{elapsed}/{run}"));
        fs::create_dir_all(leaf.join("orbax-ckpt")).unwrap();
        fs::write(leaf.join("completed"), b"").unwrap();
        fs::write(leaf.join("checksums.0.json"), b"{}").unwrap();
        leaf.to_string_lossy().into_owned()
    }
}

fn request(uri: &str, from: IpAddr, body: body::Body) -> Inner {
    let mut request = http::Request::builder().uri(uri).body(body).unwrap();
    request
        .extensions_mut()
        .insert(PeerAddr(SocketAddr::new(from, 41000)));
    request
}

async fn list(svc: &mut GatewayService<FakeCopy>, from: IpAddr) -> Vec<(String, usize)> {
    let response = svc
        .call(request(
            LIST,
            from,
            body::Body::new(freeze(get_bytes_mut())),
        ))
        .await
        .unwrap();
    let (mut all, mut p, mut s, mut sizes) = (
        Vec::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
    );
    parse(
        proto![
            (1, bytes(&mut all)),
            (2, repeated_ints(&mut p)),
            (3, repeated_ints(&mut s)),
            (4, repeated_ints(&mut sizes))
        ],
        response.into_body(),
        BodyKind::Response,
    )
    .await
    .unwrap();
    decode_names(all, p, s)
        .unwrap()
        .into_iter()
        .map(|n| String::from_utf8(n).unwrap())
        .zip(sizes)
        .collect()
}

async fn listed_prefixes(svc: &mut GatewayService<FakeCopy>, from: IpAddr) -> Vec<String> {
    let mut prefixes: Vec<String> = list(svc, from)
        .await
        .iter()
        .filter_map(|(n, _)| prefix_of(n).map(str::to_string))
        .collect();
    prefixes.dedup();
    prefixes
}

async fn send(
    svc: &mut GatewayService<FakeCopy>,
    from: IpAddr,
    name: &str,
    size: usize,
) -> Result<Vec<u8>, Status> {
    let (all, p, s) = encode_names(vec![name.as_bytes().to_vec()]);
    let mut buf = get_bytes_mut();
    buf.put_string(1, &all);
    buf.put_repeated_ints([2, 3, 4, 5], [&p, &s, &[0], &[size]]);
    let response = svc
        .call(request(SEND, from, body::Body::new(freeze(buf))))
        .await
        .unwrap();
    if let Some(status) = Status::from_header_map(response.headers()) {
        return Err(status);
    }
    let mut data = Vec::new();
    parse(
        proto![(1, bytes(&mut data))],
        response.into_body(),
        BodyKind::Response,
    )
    .await?;
    Ok(data)
}

fn file(elapsed: u64, name: &str) -> Vec<u8> {
    files(elapsed)
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1
}

async fn held_at_1000(fx: &Fixture) -> (Arc<Gateway>, GatewayService<FakeCopy>, String) {
    write_checkpoint(&fx.root, 900);
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    let mut svc = fx.service(&gateway, &fx.root);
    gateway.note_mirrored(&real(1000));
    let follow = json!({"version": 1, "mode": "follow", "released": {"run": "inc", "generation": 1, "seqOffset": 0}});
    gateway.apply(&follow, Instant::now()).unwrap();
    let serving = serving_prefix(1000, "inc", 1);
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving.clone()]
    );
    let leaf = fx.nfs_leaf(1000);
    let applied = gateway
        .apply(&held_policy(2, 1000, 1, &leaf), Instant::now())
        .unwrap();
    assert_eq!(applied.held, Some(real(1000)));
    (gateway, svc, serving)
}

#[tokio::test]
async fn a_held_gateway_lists_only_the_held_prefix_to_every_pod() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;

    write_checkpoint(&fx.root, 1100);
    gateway.note_mirrored(&real(1100));
    fs::remove_dir_all(fx.root.join("elapsed_samples_000000000000001000")).unwrap();

    let held_files: Vec<(String, usize)> = {
        let mut f: Vec<_> = files(1000)
            .into_iter()
            .map(|(n, d)| (format!("{held}/{n}"), d.len()))
            .collect();
        f.sort();
        f
    };
    for pod in [ip(1), ip(2), ip(16), ip(99)] {
        assert_eq!(list(&mut svc, pod).await, held_files, "pod {pod}");
        assert_eq!(
            send(&mut svc, pod, &format!("{held}/emb_table/c/1/0"), 200)
                .await
                .unwrap(),
            file(1000, "emb_table/c/1/0")
        );
        for refused in [
            format!("{}/emb_table/c/0/0", real(1100)),
            format!("{}/emb_table/c/0/0", serving_prefix(1100, "inc", 1)),
        ] {
            let err = send(&mut svc, pod, &refused, 200).await.unwrap_err();
            assert_eq!(err.code(), tonic::Code::NotFound, "{refused}: {err}");
        }
    }
}

#[tokio::test]
async fn a_restarted_gateway_in_hold_serves_its_restore_prefix() {
    let fx = Fixture::new();
    let (gateway, _, held) = held_at_1000(&fx).await;
    gateway.reconcile_durable().unwrap();
    drop(gateway);
    fs::remove_dir_all(fx.root.join("elapsed_samples_000000000000001000")).unwrap();
    write_checkpoint(&fx.root, 1100);

    let restarted = fx.open();
    restarted.note_mirrored(&real(1100));
    let mut svc = fx.service(&restarted, &fx.root);
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);
    assert_eq!(
        send(&mut svc, ip(1), &format!("{held}/dense_params/w"), 96)
            .await
            .unwrap(),
        file(1000, "dense_params/w")
    );

    let fresh_root = tempfile::tempdir().unwrap();
    write_checkpoint(fresh_root.path(), 1100);
    let moved = Arc::new(Gateway::open(fx.config(fresh_root.path(), 0)));
    moved.reconcile_durable().unwrap();
    moved.note_mirrored(&real(1100));
    let mut svc = fx.service(&moved, fresh_root.path());
    for pod in [ip(2), ip(99)] {
        assert_eq!(listed_prefixes(&mut svc, pod).await, vec![held.clone()]);
        assert_eq!(
            send(&mut svc, pod, &format!("{held}/emb_table/c/0/0"), 200)
                .await
                .unwrap(),
            file(1000, "emb_table/c/0/0")
        );
    }
}

#[tokio::test]
async fn a_gateway_without_a_fresh_policy_keeps_holding() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    write_checkpoint(&fx.root, 1100);
    gateway.note_mirrored(&real(1100));
    let leaf = fx.nfs_leaf(1000);

    let stale = Instant::now()
        .checked_sub(Duration::from_secs(600))
        .unwrap();
    gateway
        .apply(&held_policy(2, 1000, 1, &leaf), stale)
        .unwrap();
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);

    assert!(
        gateway
            .apply(&json!({"version": 3, "mode": "canary"}), Instant::now())
            .is_err()
    );
    let older = json!({"version": 1, "mode": "follow", "released": {"run": "x", "generation": 9, "seqOffset": 5000}});
    assert!(
        gateway
            .apply(&older, Instant::now())
            .unwrap_err()
            .contains("older")
    );
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);

    let mut canary = held_policy(3, 1000, 1, &leaf);
    canary["mode"] = json!("canary");
    canary["released"] = json!({"run": "cand", "generation": 2, "seqOffset": 0});
    canary["slots"][0]["canaryPods"] = json!(2);
    gateway.apply(&canary, stale).unwrap();
    for pod in [ip(1), ip(2), ip(3)] {
        assert_eq!(
            listed_prefixes(&mut svc, pod).await,
            vec![held.clone()],
            "pod {pod}"
        );
    }
}

fn serving_dir_entries(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = scan_prefixes(dir).into_iter().map(|(p, _)| p).collect();
    out.sort();
    out
}

#[tokio::test]
async fn the_serving_dir_never_moves_past_the_release() {
    let fx = Fixture::new();
    let (gateway, _, held) = held_at_1000(&fx).await;
    let leaf = fx.nfs_leaf(1000);
    let dir = fx.nfs.join("serving");
    fs::create_dir_all(dir.join(format!("{}/orbax-ckpt", serving_prefix(5000, "old", 1)))).unwrap();

    let mut hold = held_policy(3, 1000, 1, &leaf);
    hold["nfs"] = json!({"dir": dir, "offered": held});
    assert_eq!(
        gateway.apply(&hold, Instant::now()).unwrap().nfs_refused,
        None
    );
    assert_eq!(serving_dir_entries(&dir), vec![held.clone()]);
    let served = dir.join(&held);
    assert!(
        fs::symlink_metadata(&served).unwrap().is_dir(),
        "the leaf is a real dir"
    );
    assert_eq!(
        fs::read_link(served.join("completed")).unwrap(),
        Path::new(&leaf).join("completed")
    );
    assert!(served.join("orbax-ckpt").is_dir());

    let candidate = serving_prefix(1100, "cand", 2);
    for (version, mode, all_released) in [(4, "canary", false), (5, "rolling", false)] {
        let mut p = hold.clone();
        p["version"] = json!(version);
        p["mode"] = json!(mode);
        p["released"] = json!({"run": "cand", "generation": 2, "seqOffset": 600});
        p["slots"][0]["released"] = json!(all_released);
        p["nfs"]["offered"] = json!(candidate);
        let refused = gateway
            .apply(&p, Instant::now())
            .unwrap()
            .nfs_refused
            .unwrap();
        assert!(refused.contains("sorts above"), "{refused}");
        assert_eq!(serving_dir_entries(&dir), vec![held.clone()]);
    }

    let rollback = serving_prefix(1200, "inc", 3);
    let mut back = held_policy(6, 1200, 3, &leaf);
    back["nfs"] = json!({"dir": dir, "offered": rollback});
    assert_eq!(
        gateway.apply(&back, Instant::now()).unwrap().nfs_refused,
        None
    );
    assert_eq!(serving_dir_entries(&dir), vec![rollback]);
}

async fn canary_of_500(fx: &Fixture, gateway: &Gateway, version: u64, seq_offset: i64) -> Value {
    assert!(
        gateway.rewind_to(&real_of(500, CAND)),
        "the trainer's newest sorts below the root's"
    );
    assert!(!fx.root.join("elapsed_samples_000000000000001000").exists());
    assert!(!fx.root.join("elapsed_samples_000000000000000900").exists());
    write_checkpoint_of(&fx.root, 500, CAND);
    gateway.note_mirrored(&real_of(500, CAND));
    let mut canary = held_policy(version, 1000, 1, &fx.nfs_leaf(1000));
    canary["mode"] = json!("canary");
    canary["released"] = json!({"run": "cand", "generation": 2, "seqOffset": seq_offset});
    canary["slots"][0]["canaryPods"] = json!(2);
    gateway.apply(&canary, Instant::now()).unwrap();
    canary
}

#[tokio::test]
async fn the_gateway_admits_at_most_the_planned_pods() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    let canary = canary_of_500(&fx, &gateway, 3, 600).await;
    let candidate = serving_prefix(1100, "cand", 2);

    let mut admitted = Vec::new();
    for round in 0..3 {
        for n in [1, 2, 3, 4, 5, 16, 99] {
            let got = listed_prefixes(&mut svc, ip(n)).await;
            if got == vec![candidate.clone()] {
                if round == 0 {
                    admitted.push(n);
                }
            } else {
                assert_eq!(got, vec![held.clone()], "pod {n}");
            }
        }
        assert_eq!(admitted, vec![1, 2], "round {round}");
    }
    assert_eq!(listed_prefixes(&mut svc, ip(16)).await, vec![held.clone()]);
    assert_eq!(
        send(&mut svc, ip(3), &format!("{candidate}/dense_params/w"), 96)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::NotFound
    );
    assert_eq!(
        send(&mut svc, ip(1), &format!("{candidate}/dense_params/w"), 96)
            .await
            .unwrap(),
        file(500, "dense_params/w")
    );

    let ack = gateway.ack().unwrap();
    assert_eq!(ack["offered"]["uid-1"], json!(candidate));
    assert_eq!(ack["offered"]["uid-3"], json!(held));
    let replica = Arc::new(Gateway::open(GatewayConfig {
        state_dir: None,
        ..fx.config(&fx.root, 1)
    }));
    replica.note_mirrored(&real_of(500, CAND));
    replica.apply(&canary, Instant::now()).unwrap();
    let mut other = fx.service(&replica, &fx.root);
    assert_eq!(listed_prefixes(&mut other, ip(1)).await, vec![held.clone()]);
    let mut written_back = canary.clone();
    written_back["version"] = json!(4);
    written_back["slots"][0]["admitted"] = json!(["uid-1", "uid-2"]);
    replica.apply(&written_back, Instant::now()).unwrap();
    for (n, want) in [(1, &candidate), (2, &candidate), (3, &held), (4, &held)] {
        assert_eq!(
            &listed_prefixes(&mut other, ip(n)).await,
            &vec![want.clone()],
            "pod {n}"
        );
    }
}

#[tokio::test]
async fn a_release_with_fewer_samples_is_offered_above_what_pods_serve() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;

    let offered = listed_prefixes(&mut svc, ip(1)).await;
    assert_eq!(offered, vec![serving_prefix(1100, "cand", 2)]);
    let serving_now = format!("elapsed_samples_{:018}/", 1000);
    assert!(format!("{}/", offered[0]).as_str() > serving_now.as_str());
    assert!(offered[0] > held);

    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 500).await;
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held]);
}

#[tokio::test]
async fn a_rollback_is_offered_above_the_candidate() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;
    let candidate = serving_prefix(1100, "cand", 2);
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![candidate.clone()]
    );
    assert_eq!(
        listed_prefixes(&mut svc, ip(2)).await,
        vec![candidate.clone()]
    );
    assert_eq!(listed_prefixes(&mut svc, ip(3)).await, vec![held.clone()]);

    let rollback = serving_prefix(1200, "inc", 3);
    gateway
        .apply(&held_policy(4, 1200, 3, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    for pod in [ip(1), ip(3), ip(99)] {
        let got = listed_prefixes(&mut svc, pod).await;
        assert_eq!(got, vec![rollback.clone()], "pod {pod}");
        assert!(got[0] > candidate && got[0] > held);
        assert_eq!(
            send(&mut svc, pod, &format!("{rollback}/emb_table/c/0/0"), 200)
                .await
                .unwrap(),
            file(1000, "emb_table/c/0/0"),
            "the rollback serves the restore checkpoint's bytes"
        );
    }
    assert_eq!(gateway.ack().unwrap()["servingSeqMax"], json!(1200));
}

#[tokio::test]
async fn a_finished_follow_keeps_the_restore_checkpoint_for_a_rollback_after_it() {
    let fx = Fixture::new();
    let (gateway, mut svc, _) = held_at_1000(&fx).await;
    gateway.reconcile_durable().unwrap();
    canary_of_500(&fx, &gateway, 3, 600).await;
    let leaf = fx.nfs_leaf(1000);
    let mut finished = held_policy(4, 1000, 1, &leaf);
    finished["mode"] = json!("follow");
    finished["released"] = json!({"run": "cand", "generation": 2, "seqOffset": 600});
    gateway.apply(&finished, Instant::now()).unwrap();
    gateway.reconcile_durable().unwrap();

    let candidate = serving_prefix(1100, "cand", 2);
    for pod in [ip(1), ip(3), ip(99)] {
        assert_eq!(
            listed_prefixes(&mut svc, pod).await,
            vec![candidate.clone()],
            "pod {pod}"
        );
    }
    assert_eq!(gateway.ack().unwrap()["heldPinned"], json!(true));
    assert_eq!(
        scan_prefixes(&held_root_of(&fx.root)),
        vec![(real(1000), file_sizes(1000))]
    );
    assert_eq!(
        scan_prefixes(&fx.state.join("s0").join(HELD_DIR)),
        vec![(real(1000), file_sizes(1000))]
    );

    let rollback = serving_prefix(1200, "inc", 3);
    gateway
        .apply(&held_policy(5, 1200, 3, &leaf), Instant::now())
        .unwrap();
    assert_eq!(gateway.ack().unwrap()["heldPinned"], json!(true));
    for pod in [ip(1), ip(3), ip(99)] {
        assert_eq!(listed_prefixes(&mut svc, pod).await, vec![rollback.clone()]);
        assert_eq!(
            send(&mut svc, pod, &format!("{rollback}/dense_params/w"), 96)
                .await
                .unwrap(),
            file(1000, "dense_params/w")
        );
    }
}

#[tokio::test]
async fn a_follow_moves_the_nfs_serving_dir_with_what_pods_are_offered() {
    let fx = Fixture::new();
    let (gateway, mut svc, _) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;
    let dir = fx.nfs.join("serving");
    let leaf = fx.nfs_leaf(1000);
    let offered = serving_prefix(1100, "cand", 2);
    let mut finished = held_policy(4, 1000, 1, &leaf);
    finished["mode"] = json!("follow");
    finished["released"] = json!({"run": "cand", "generation": 2, "seqOffset": 600});
    finished["nfs"] =
        json!({"dir": dir, "offered": offered, "offeredPath": fx.nfs_leaf_of(500, CAND)});
    assert_eq!(
        gateway
            .apply(&finished, Instant::now())
            .unwrap()
            .nfs_refused,
        None
    );
    assert_eq!(serving_dir_entries(&dir), vec![offered]);

    write_checkpoint_of(&fx.root, 700, CAND);
    gateway.note_mirrored(&real_of(700, CAND));
    let next = serving_prefix(1300, "cand", 2);
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![next.clone()]);
    let candidate_leaf = fx.nfs_leaf_of(700, CAND);
    assert_eq!(
        gateway
            .apply(&finished, Instant::now())
            .unwrap()
            .nfs_refused,
        None
    );
    assert_eq!(serving_dir_entries(&dir), vec![next.clone()]);
    assert_eq!(
        fs::read_link(dir.join(&next).join("completed")).unwrap(),
        Path::new(&candidate_leaf).join("completed")
    );

    write_checkpoint_of(&fx.root, 800, CAND);
    gateway.note_mirrored(&real_of(800, CAND));
    let refused = gateway
        .apply(&finished, Instant::now())
        .unwrap()
        .nfs_refused
        .expect("800 is not on NFS yet");
    assert!(refused.contains("elapsed_samples_800"), "{refused}");
    assert_eq!(serving_dir_entries(&dir), vec![next]);

    let held = serving_prefix(2000, "inc", 3);
    let mut follow = held_policy(5, 2000, 3, &leaf);
    follow["nfs"] = json!({"dir": dir, "offered": held});
    gateway.apply(&follow, Instant::now()).unwrap();
    assert_eq!(serving_dir_entries(&dir), vec![held.clone()]);
    follow["version"] = json!(6);
    follow["mode"] = json!("follow");
    follow["released"] =
        json!({"run": "inc", "generation": 3, "seqOffset": 1000, "from": real_of(1100, INC)});
    gateway.apply(&follow, Instant::now()).unwrap();
    assert_eq!(serving_dir_entries(&dir), vec![held]);
    write_checkpoint_of(&fx.root, 1100, INC);
    gateway.note_mirrored(&real_of(1100, INC));
    fx.nfs_leaf_of(1100, INC);
    assert_eq!(
        gateway.apply(&follow, Instant::now()).unwrap().nfs_refused,
        None
    );
    assert_eq!(
        serving_dir_entries(&dir),
        vec![serving_prefix(2100, "inc", 3)]
    );
}

fn served_follow(version: u64, dir: &Path, offered_path: Option<String>) -> Value {
    let mut follow = held_policy(version, 1000, 1, "");
    follow["mode"] = json!("follow");
    follow["held"] = Value::Null;
    follow["released"] = json!({"run": "cand", "generation": 2, "seqOffset": 600});
    follow["nfs"] = json!({"dir": dir});
    if let Some(path) = offered_path {
        follow["nfs"]["offeredPath"] = json!(path);
    }
    follow
}

#[tokio::test]
async fn a_follow_as_the_control_plane_serves_it_moves_the_nfs_dir_or_refuses_nothing() {
    let fx = Fixture::new();
    let (gateway, _svc, _) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;
    let dir = fx.nfs.join("serving");
    let before = serving_dir_entries(&dir);
    let plain = served_follow(4, &dir, None);
    assert_eq!(
        gateway.apply(&plain, Instant::now()).unwrap().nfs_refused,
        None,
        "a follow naming no NFS checkpoint refuses nothing"
    );
    assert_eq!(serving_dir_entries(&dir), before);

    let finished = served_follow(5, &dir, Some(fx.nfs_leaf_of(500, CAND)));
    assert_eq!(
        gateway
            .apply(&finished, Instant::now())
            .unwrap()
            .nfs_refused,
        None
    );
    assert_eq!(
        serving_dir_entries(&dir),
        vec![serving_prefix(1100, "cand", 2)]
    );
}

#[tokio::test]
async fn a_follow_moves_the_nfs_serving_dir_only_to_a_finished_save() {
    let fx = Fixture::new();
    let (gateway, _svc, _) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;
    let dir = fx.nfs.join("serving");
    let follow = served_follow(4, &dir, Some(fx.nfs_leaf_of(500, CAND)));
    assert_eq!(
        gateway.apply(&follow, Instant::now()).unwrap().nfs_refused,
        None
    );
    let offered = serving_prefix(1100, "cand", 2);
    assert_eq!(serving_dir_entries(&dir), vec![offered.clone()]);

    write_checkpoint_of(&fx.root, 700, CAND);
    gateway.note_mirrored(&real_of(700, CAND));
    let leaf = fx.nfs.join("ckpt/elapsed_samples_700").join(CAND);
    let tmp = leaf.join("orbax-ckpt.orbax-checkpoint-tmp-1700000000");
    fs::create_dir_all(&tmp).unwrap();
    fs::write(leaf.join("checksums.0.json"), b"{}").unwrap();
    let refused = gateway
        .apply(&follow, Instant::now())
        .unwrap()
        .nfs_refused
        .expect("700 is still being written on NFS");
    assert!(refused.contains("elapsed_samples_700"), "{refused}");
    assert_eq!(serving_dir_entries(&dir), vec![offered]);

    fs::rename(&tmp, leaf.join("orbax-ckpt")).unwrap();
    fs::write(leaf.join("completed"), b"").unwrap();
    assert_eq!(
        gateway.apply(&follow, Instant::now()).unwrap().nfs_refused,
        None
    );
    assert_eq!(
        serving_dir_entries(&dir),
        vec![serving_prefix(1300, "cand", 2)]
    );
}

#[tokio::test]
async fn a_rolled_back_gateway_offers_no_candidate_checkpoint_again() {
    let fx = Fixture::new();
    let (gateway, mut svc, _) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;
    write_checkpoint_of(&fx.root, 1300, CAND);
    gateway.note_mirrored(&real_of(1300, CAND));
    let leaf = fx.nfs_leaf(1000);
    let held = serving_prefix(2000, "inc", 3);
    gateway
        .apply(&held_policy(4, 2000, 3, &leaf), Instant::now())
        .unwrap();
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);

    let mut follow = held_policy(5, 2000, 3, &leaf);
    follow["mode"] = json!("follow");
    follow["released"] =
        json!({"run": "inc", "generation": 3, "seqOffset": 1000, "from": real_of(1100, INC)});
    gateway.apply(&follow, Instant::now()).unwrap();
    for pod in [ip(1), ip(4), ip(99)] {
        assert_eq!(
            listed_prefixes(&mut svc, pod).await,
            vec![held.clone()],
            "pod {pod} was offered the rolled-back candidate's checkpoint"
        );
        let relabelled = format!("{}/dense_params/w", serving_prefix(2300, "inc", 3));
        let err = send(&mut svc, pod, &relabelled, 96).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound, "pod {pod}");
    }

    assert!(gateway.rewind_to(&real_of(1100, INC)));
    write_checkpoint_of(&fx.root, 1100, INC);
    gateway.note_mirrored(&real_of(1100, INC));
    let first = serving_prefix(2100, "inc", 3);
    for pod in [ip(1), ip(4)] {
        assert_eq!(listed_prefixes(&mut svc, pod).await, vec![first.clone()]);
        assert_eq!(
            send(&mut svc, pod, &format!("{first}/dense_params/w"), 96)
                .await
                .unwrap(),
            file(1100, "dense_params/w")
        );
    }
    write_checkpoint_of(&fx.root, 1200, INC);
    gateway.note_mirrored(&real_of(1200, INC));
    let next = serving_prefix(2200, "inc", 3);
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![next.clone()]);

    gateway.reconcile_durable().unwrap();
    drop(gateway);
    let restarted = fx.open();
    restarted.apply(&follow, Instant::now()).unwrap();
    let mut svc = fx.service(&restarted, &fx.root);
    assert_eq!(listed_prefixes(&mut svc, ip(5)).await, vec![next]);

    let mut missed = follow.clone();
    missed["version"] = json!(6);
    missed["released"]["from"] = json!(real_of(1250, INC));
    let probed_before = restarted.probe_mark();
    restarted.apply(&missed, Instant::now()).unwrap();
    assert_eq!(listed_prefixes(&mut svc, ip(6)).await, vec![held.clone()]);
    write_checkpoint_of(&fx.root, 1400, CAND);
    restarted.note_mirrored_probed(&real_of(1400, CAND), probed_before);
    assert_eq!(
        listed_prefixes(&mut svc, ip(6)).await,
        vec![held.clone()],
        "a candidate download begun before `from` was named counted as the restored trainer's"
    );
    assert!(restarted.rewind_to(&real_of(1300, INC)));
    write_checkpoint_of(&fx.root, 1300, INC);
    restarted.note_mirrored(&real_of(1300, INC));
    assert_eq!(
        listed_prefixes(&mut svc, ip(6)).await,
        vec![serving_prefix(2300, "inc", 3)]
    );
}

#[tokio::test]
async fn a_follow_whose_from_the_trainer_already_passed_follows_on_the_next_poll() {
    let fx = Fixture::new();
    let (gateway, mut svc, _) = held_at_1000(&fx).await;
    let leaf = fx.nfs_leaf(1000);
    let held = serving_prefix(2000, "inc", 3);
    gateway
        .apply(&held_policy(3, 2000, 3, &leaf), Instant::now())
        .unwrap();
    write_checkpoint_of(&fx.root, 1100, INC);
    gateway.note_mirrored(&real_of(1100, INC));
    write_checkpoint_of(&fx.root, 1200, INC);
    gateway.note_mirrored(&real_of(1200, INC));

    let mut follow = held_policy(4, 2000, 3, &leaf);
    follow["mode"] = json!("follow");
    follow["released"] =
        json!({"run": "inc", "generation": 3, "seqOffset": 1000, "from": real_of(1100, INC)});
    gateway.apply(&follow, Instant::now()).unwrap();
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![held],
        "a newest mirrored before `from` was named counted as the restored trainer's"
    );

    gateway.note_mirrored(&real_of(1200, INC));
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving_prefix(2200, "inc", 3)],
        "the poll after the follow left serving on the hold until the trainer writes again"
    );
}

#[tokio::test]
async fn checksums_stay_consistent_after_relabel() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 500);
    let gateway = fx.open();
    gateway.note_mirrored(&real(500));
    let follow = json!({"version": 1, "mode": "follow", "released": {"run": "cand", "generation": 2, "seqOffset": 600}});
    gateway.apply(&follow, Instant::now()).unwrap();
    let mut svc = fx.service(&gateway, &fx.root);
    let serving = serving_prefix(1100, "cand", 2);

    let listed = list(&mut svc, ip(1)).await;
    let mut want: Vec<(String, usize)> = files(500)
        .into_iter()
        .map(|(n, d)| (format!("{serving}/{n}"), d.len()))
        .collect();
    want.sort();
    assert_eq!(
        listed, want,
        "same files, same sizes, under the serving name"
    );
    for (name, data) in files(500) {
        let got = send(&mut svc, ip(1), &format!("{serving}/{name}"), data.len())
            .await
            .unwrap();
        assert_eq!(
            simd_adler32::adler32(&got.as_slice()),
            simd_adler32::adler32(&data.as_slice()),
            "{name}"
        );
        assert_eq!(got, data, "{name}");
    }
}

#[test]
fn no_policy_flag_is_todays_proxy() {
    #[derive(Parser)]
    struct Server {
        #[clap(long)]
        copy_url: String,
        #[clap(long)]
        num_servers: usize,
        #[command(flatten)]
        gateway: GatewayArgs,
    }
    let today = [
        "checkpoint-server",
        "--copy-url",
        "trainer.recsys.svc:9988",
        "--num-servers",
        "1",
    ];
    let parsed = Server::try_parse_from(today).unwrap();
    assert!(!parsed.gateway.enabled());
    assert!(parsed.gateway.tls().unwrap().is_none());
    for flag in [
        ["--policy-fresh-secs", "5"],
        ["--policy-poll-secs", "5"],
        ["--policy-tls-ca", "/ca.pem"],
        ["--policy-tls-server-name", "cp"],
        ["--gateway-state-dir", "/nfs/x"],
        ["--pod-refresh-secs", "5"],
        ["--pod-stale-secs", "5"],
        ["--unknown-peer-secs", "5"],
    ] {
        let args: Vec<&str> = today.iter().copied().chain(flag).collect();
        assert!(
            Server::try_parse_from(&args).is_err(),
            "{flag:?} alone must not change the proxy"
        );
    }
    let on: Vec<&str> = today
        .iter()
        .copied()
        .chain([
            "--policy-url",
            "https://cp:8443/api/prod-trainers/7/serving-policy",
            "--policy-tls-ca",
            "/ca.pem",
        ])
        .collect();
    let parsed = Server::try_parse_from(&on).unwrap();
    assert!(parsed.gateway.enabled());
    assert_eq!(parsed.gateway.policy_fresh_secs, 120);
    assert_eq!(parsed.gateway.tls().unwrap().unwrap().ca_path, "/ca.pem");
}

type Seen = (String, String, Option<String>, String);

#[derive(Clone, Default)]
struct FakeApi {
    seen: Arc<Mutex<Vec<Seen>>>,
    deny_b: Arc<std::sync::atomic::AtomicBool>,
}

impl NamedService for FakeApi {
    const NAME: &'static str = "api";
}

fn pod(uid: &str, phase: &str, ip: &str) -> Value {
    json!({"metadata": {"uid": uid}, "status": {"phase": phase, "podIP": ip, "podIPs": [{"ip": ip}]}})
}

impl Service<Inner> for FakeApi {
    type Response = http::Response<body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Infallible>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Inner) -> Self::Future {
        let seen = self.seen.clone();
        let deny_b = self.deny_b.load(std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            let method = request.method().to_string();
            let target = request
                .uri()
                .path_and_query()
                .map(ToString::to_string)
                .unwrap_or_default();
            let auth = request
                .headers()
                .get(http::header::AUTHORIZATION)
                .map(|v| v.to_str().unwrap().to_string());
            let body = http_body_util::BodyExt::collect(request.into_body())
                .await
                .unwrap()
                .to_bytes();
            seen.lock().push((
                method,
                target.clone(),
                auth,
                String::from_utf8_lossy(&body).into_owned(),
            ));
            let (status, answer) = if target.starts_with("/api/v1/namespaces/inference/pods?") {
                let items = vec![
                    pod("uid-a", "Running", "10.1.0.1"),
                    pod("uid-b", "Pending", "10.1.0.2"),
                    pod("uid-c", "Succeeded", "10.1.0.3"),
                    pod("uid-d", "Running", "10.1.0.9"),
                    pod("uid-e", "Running", "10.1.0.9"),
                ];
                (200, json!({"kind": "PodList", "items": items}))
            } else if target.starts_with("/api/v1/namespaces/inference-b/pods?") && !deny_b {
                let items = vec![pod("uid-f", "Running", "10.1.0.20")];
                (200, json!({"kind": "PodList", "items": items}))
            } else if target.starts_with("/api/v1/namespaces/bridges/pods?") {
                let items = vec![pod("uid-bridge", "Running", "10.0.0.30")];
                (200, json!({"kind": "PodList", "items": items}))
            } else if target.starts_with("/api/v1/namespaces/") {
                (403, json!({"kind": "Status", "reason": "Forbidden"}))
            } else if target == "/api/prod-trainers/7/serving-policy" {
                (200, held_policy(3, 1000, 1, "/nfs/x"))
            } else if target == "/api/prod-trainers/7/serving-policy/ack" {
                (200, json!({}))
            } else {
                (404, json!({"error": "no such route"}))
            };
            Ok(http::Response::builder()
                .status(status)
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(body::Body::new(http_body_util::Full::new(
                    bytes::Bytes::from(answer.to_string()),
                )))
                .unwrap())
        })
    }
}

async fn serve_api() -> (String, FakeApi) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let api = FakeApi::default();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(api.clone())
            .serve_with_incoming(tonic::transport::server::TcpIncoming::from(listener)),
    );
    (origin, api)
}

#[tokio::test]
async fn the_policy_is_read_and_acked_over_http2() {
    let (origin, api) = serve_api().await;
    let client = PolicyClient::new(
        &format!("{origin}/api/prod-trainers/7/serving-policy/"),
        None,
    )
    .unwrap();
    assert_eq!(
        client.fetch().await.unwrap(),
        held_policy(3, 1000, 1, "/nfs/x")
    );
    client.ack(json!({"version": 3})).await.unwrap();
    let missing = PolicyClient::new(
        &format!("{origin}/api/prod-trainers/8/serving-policy"),
        None,
    )
    .unwrap()
    .fetch()
    .await
    .unwrap_err();
    assert!(missing.contains("404"), "{missing}");
    let seen = api.seen.lock().clone();
    assert_eq!(
        seen[..2],
        [
            (
                "GET".into(),
                "/api/prod-trainers/7/serving-policy".into(),
                None,
                String::new()
            ),
            (
                "POST".into(),
                "/api/prod-trainers/7/serving-policy/ack".into(),
                None,
                r#"{"version":3}"#.into()
            ),
        ]
    );
}

#[tokio::test]
async fn the_slots_pods_are_read_from_the_kubernetes_api() {
    let (origin, api) = serve_api().await;
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    fs::write(&token, "tok-123\n").unwrap();
    let pods = KubePods::new(&origin, None, Some(token)).unwrap();
    let mut raw = held_policy(3, 1000, 1, "/nfs/x");
    raw["slots"][0]["podSelector"] = json!({"matchLabels": {"tier": "a", "app": "replies"}});
    let good = raw.clone();
    let policy = Policy::from_json(&raw).unwrap();
    assert_eq!(pods.refresh(&policy).await.unwrap(), 2);
    let at = |s: &str| pods.lookup(s.parse().unwrap());
    assert_eq!(at("10.1.0.1"), Some((30, "uid-a".into())));
    assert_eq!(at("10.1.0.2"), Some((30, "uid-b".into())));
    assert_eq!(
        at("10.1.0.3"),
        None,
        "a finished pod's address may be reused"
    );
    assert_eq!(at("10.1.0.9"), None, "an address two pods share is no pod");
    assert_eq!(
        api.seen.lock()[0],
        (
            "GET".into(),
            "/api/v1/namespaces/inference/pods?labelSelector=app%3Dreplies%2Ctier%3Da".into(),
            Some("Bearer tok-123".into()),
            String::new()
        )
    );

    for selector in [
        json!({}),
        json!(""),
        json!(7),
        json!({"matchExpressions": []}),
    ] {
        let mut raw = raw.clone();
        raw["slots"][0]["podSelector"] = selector.clone();
        let policy = Policy::from_json(&raw).unwrap();
        assert!(
            pods.refresh(&policy).await.is_err(),
            "{selector} would list every pod in the namespace"
        );
        assert_eq!(
            at("10.1.0.1"),
            None,
            "{selector}: the slot's pods are no longer known, so they are held"
        );
        assert_eq!(
            pods.refresh(&Policy::from_json(&good).unwrap()).await,
            Ok(2)
        );
    }
}

#[test]
fn a_rewound_gateway_fetches_only_what_its_root_lacks() {
    let fx = Fixture::new();
    for elapsed in [500, 1000] {
        write_checkpoint(&fx.root, elapsed);
    }
    let other_run = fx
        .root
        .join("elapsed_samples_000000000000000700/run-old/dense_params/w");
    fs::create_dir_all(other_run.parent().unwrap()).unwrap();
    fs::write(&other_run, b"old").unwrap();
    let gateway = fx.open();

    assert!(gateway.rewind_to(&real(700)));
    assert!(
        !fx.root.join("elapsed_samples_000000000000000700").exists(),
        "the proxy renames onto elapsed_samples_<N>/, which another run's dir there would refuse"
    );
    assert!(!fx.root.join(real(1000)).exists());
    assert!(fx.root.join(real(500)).is_dir());
    assert!(!gateway.mirrors(&real(700)), "the fresh proxy fetches 700");

    write_checkpoint(&fx.root, 1000);
    assert!(gateway.rewind_to(&real(500)));
    assert!(
        gateway.mirrors(&real(500)),
        "the fresh proxy must not fetch 500 again: its rename onto the root's 500 fails every poll"
    );
    assert!(!gateway.rewind_to(&real(500)), "nothing left above 500");
}

fn held_root_of(root: &Path) -> PathBuf {
    root.join(GATEWAY_DIR).join(HELD_DIR)
}

fn file_sizes(elapsed: u64) -> Vec<(String, usize)> {
    let mut out: Vec<_> = files(elapsed)
        .into_iter()
        .map(|(n, d)| (n.to_string(), d.len()))
        .collect();
    out.sort();
    out
}

fn follow_inc(version: u64) -> Value {
    json!({"version": version, "mode": "follow", "released": {"run": "inc", "generation": 1, "seqOffset": 0}})
}

#[derive(Clone)]
struct SilentTrainer;

impl NamedService for SilentTrainer {
    const NAME: &'static str = "copy.Copy";
}

impl Service<Inner> for SilentTrainer {
    type Response = http::Response<body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Infallible>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: Inner) -> Self::Future {
        Box::pin(futures::future::pending())
    }
}

#[tokio::test]
async fn a_trainer_that_never_answers_does_not_stall_the_poll() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(SilentTrainer)
            .serve_with_incoming(tonic::transport::server::TcpIncoming::from(listener)),
    );
    let newest = tokio::time::timeout(
        Duration::from_secs(20),
        trainer_newest(&url, Duration::from_millis(500)),
    )
    .await
    .expect("the trainer probe never returned, so the proxy poll and the mirror never run");
    assert_eq!(newest, None);
}

#[tokio::test]
async fn a_new_hold_takes_effect_before_its_durable_copy() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    gateway.apply(&follow_inc(1), Instant::now()).unwrap();
    let durable = fx.state.join("s0");
    let _ = fs::remove_dir_all(&durable);

    let applied = gateway
        .apply(&held_policy(2, 1000, 1, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    assert_eq!(applied.held, Some(real(1000)));
    assert_eq!(gateway.policy().unwrap().mode, Mode::Hold);
    assert!(
        !durable.exists(),
        "the policy loop waited on the state dir (NFS): the hold took effect only after the copy"
    );
    let mut svc = fx.service(&gateway, &fx.root);
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving_prefix(1000, "inc", 1)]
    );

    gateway.reconcile_durable().unwrap();
    let copied = scan_prefixes(&durable.join(HELD_DIR));
    assert_eq!(copied, vec![(real(1000), file_sizes(1000))]);
    assert!(durable.join(STATE_FILE).is_file());
}

#[tokio::test]
async fn another_runs_checkpoint_at_the_held_elapsed_is_never_held() {
    let fx = Fixture::new();
    let candidate = "elapsed_samples_000000000000001000/cand-run";
    for (name, data) in files(5) {
        let path = fx.root.join(candidate).join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, data).unwrap();
    }
    let gateway = fx.open();
    gateway.note_mirrored(candidate);
    gateway.apply(&follow_inc(1), Instant::now()).unwrap();

    let applied = gateway
        .apply(&held_policy(2, 2000, 3, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    assert_eq!(
        applied.held, None,
        "the candidate's checkpoint was held as the rollback"
    );
    let mut svc = fx.service(&gateway, &fx.root);
    for pod in [ip(1), ip(99)] {
        assert_eq!(list(&mut svc, pod).await, vec![], "pod {pod}");
    }

    write_checkpoint(&fx.root, 1000);
    let applied = gateway
        .apply(&held_policy(3, 2000, 3, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    assert_eq!(applied.held, Some(real(1000)));
    let cand_leaf = fx.nfs.join("ckpt/elapsed_samples_1000/cand-run");
    fs::create_dir_all(&cand_leaf).unwrap();
    let applied = gateway
        .apply(
            &held_policy(4, 2100, 4, &cand_leaf.to_string_lossy()),
            Instant::now(),
        )
        .unwrap();
    assert_eq!(applied.held, Some(candidate.to_string()));
    let mut svc = fx.service(&gateway, &fx.root);
    let held = serving_prefix(2100, "inc", 4);
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);
    assert_eq!(
        send(&mut svc, ip(1), &format!("{held}/dense_params/w"), 96)
            .await
            .unwrap(),
        file(5, "dense_params/w")
    );
}

#[test]
fn the_durable_copy_is_retried_until_it_is_made() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    gateway.apply(&follow_inc(1), Instant::now()).unwrap();
    let durable = fx.state.join("s0");
    let copy = durable.join(HELD_DIR);
    fs::create_dir_all(&durable).unwrap();
    fs::write(&copy, b"").unwrap();
    let hold = held_policy(2, 1000, 1, &fx.nfs_leaf(1000));
    gateway.apply(&hold, Instant::now()).unwrap();
    let _ = gateway.reconcile_durable();

    fs::remove_file(&copy).unwrap();
    let leftover = copy.join(".elapsed_samples_000000000000001000");
    fs::create_dir_all(leftover.join(RUN)).unwrap();
    fs::write(leftover.join(RUN).join("checksums.0.json"), b"{").unwrap();
    gateway.apply(&hold, Instant::now()).unwrap();
    let _ = gateway.reconcile_durable();
    assert_eq!(
        scan_prefixes(&copy),
        vec![(real(1000), file_sizes(1000))],
        "the held checkpoint has no durable copy after the first copy failed"
    );
    assert!(!leftover.exists(), "the killed copy's staging dir stays");
}

#[tokio::test]
async fn one_bad_slot_leaves_the_other_slots_pods_known() {
    let (origin, _api) = serve_api().await;
    let pods = KubePods::new(&origin, None, None).unwrap();
    let mut raw = held_policy(3, 1000, 1, "/nfs/x");
    let slot = |id: i64, namespace: &str, selector: Value| json!({"slotId": id, "namespace": namespace, "podSelector": selector, "released": false, "admitted": []});
    raw["slots"] = json!([
        slot(30, "inference", json!("app=replies")),
        slot(
            31,
            "inference-b",
            json!({"matchExpressions": [{"key": "app"}]})
        ),
        slot(32, "forbidden", json!("app=replies")),
    ]);
    let policy = Policy::from_json(&raw).unwrap();
    let err = pods.refresh(&policy).await.unwrap_err();
    let at = |s: &str| pods.lookup(s.parse().unwrap());
    assert_eq!(
        at("10.1.0.1"),
        Some((30, "uid-a".into())),
        "a bad slot made every slot's pods unknown"
    );
    assert_eq!(at("10.1.0.20"), None);
    assert!(err.contains("slot 31") && err.contains("slot 32"), "{err}");
}

#[tokio::test]
async fn a_slot_the_gateway_cannot_list_is_named_in_its_ack() {
    let fx = Fixture::new();
    let gateway = fx.open();
    let (origin, api) = serve_api().await;
    let pods = KubePods::new(&origin, None, None).unwrap();
    let named = |version: u64, selector: Value| {
        let mut hold = held_policy(version, 1000, 1, &fx.nfs_leaf(1000));
        hold["slots"][0]["podSelector"] = selector;
        hold
    };
    let selector = "app=recsys-inference-replies-experiment-1";
    gateway
        .apply(&named(2, json!(selector)), Instant::now())
        .unwrap();
    assert_eq!(pod_watch_once(&gateway, &pods).await, Some(Ok(2)));
    assert_eq!(
        api.seen.lock().last().unwrap().1,
        "/api/v1/namespaces/inference/pods?labelSelector=app%3Drecsys-inference-replies-experiment-1"
    );
    assert_eq!(
        pods.lookup("10.1.0.1".parse().unwrap()),
        Some((30, "uid-a".into()))
    );
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["podsRefused"], Value::Null, "{ack}");

    gateway.apply(&named(3, json!("")), Instant::now()).unwrap();
    let refused = pod_watch_once(&gateway, &pods).await.unwrap().unwrap_err();
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["podsRefused"], json!(refused), "{ack}");
    assert!(
        refused.contains("slot 30: podSelector \"\" is not a non-empty label selector"),
        "{refused}"
    );

    gateway
        .apply(&named(4, json!(selector)), Instant::now())
        .unwrap();
    assert_eq!(pod_watch_once(&gateway, &pods).await, Some(Ok(2)));
    assert_eq!(gateway.ack().unwrap()["podsRefused"], Value::Null);
}

#[tokio::test]
async fn a_slots_bridge_is_counted_as_a_bridge_never_as_an_unknown_peer() {
    let fx = Fixture::new();
    let (gateway, _, held) = held_at_1000(&fx).await;
    let (origin, api) = serve_api().await;
    let pods = Arc::new(KubePods::new(&origin, None, None).unwrap());
    let mut policy = held_policy(3, 1000, 1, &fx.nfs_leaf(1000));
    policy["slots"][0]["bridge"] =
        json!({"namespace": "bridges", "podSelector": "app=checkpoint-server-replies"});
    gateway.apply(&policy, Instant::now()).unwrap();
    let mut svc = GatewayService::new(
        FakeCopy {
            root: fx.root.clone(),
            fail_sends: false,
        },
        gateway.clone(),
        pods.clone(),
    );
    assert!(
        matches!(pod_watch_once(&gateway, &pods).await, Some(Ok(_))),
        "{:?}",
        api.seen.lock()
    );
    assert!(
        api.seen.lock().iter().any(|(_, target, _, _)| target
            == "/api/v1/namespaces/bridges/pods?labelSelector=app%3Dcheckpoint-server-replies"),
        "{:?}",
        api.seen.lock()
    );
    let before = gateway.ack().unwrap()["unknownPeers"].as_u64().unwrap();

    assert_eq!(listed_prefixes(&mut svc, ip(30)).await, vec![held.clone()]);
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["unknownPeers"], json!(before), "{ack}");
    assert_eq!(ack["bridgePeers"], json!(1), "{ack}");

    assert_eq!(listed_prefixes(&mut svc, ip(99)).await, vec![held]);
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["unknownPeers"], json!(before + 1), "{ack}");
    assert_eq!(ack["bridgePeers"], json!(1), "{ack}");
}

#[tokio::test]
async fn a_slot_whose_last_listing_still_serves_is_not_refused() {
    let fx = Fixture::new();
    let gateway = fx.open();
    let (origin, api) = serve_api().await;
    let mut policy = held_policy(2, 1000, 1, &fx.nfs_leaf(1000));
    policy["slots"]
        .as_array_mut()
        .unwrap()
        .push(json!({"slotId": 31, "namespace": "inference-b", "podSelector": "app=replies"}));
    gateway.apply(&policy, Instant::now()).unwrap();
    let pods = KubePods::new(&origin, None, None).unwrap();
    assert_eq!(pod_watch_once(&gateway, &pods).await, Some(Ok(3)));

    api.deny_b.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(matches!(
        pod_watch_once(&gateway, &pods).await,
        Some(Err(e)) if e.contains("slot 31")
    ));
    assert_eq!(
        pods.lookup("10.1.0.20".parse().unwrap()),
        Some((31, "uid-f".into()))
    );
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["podsRefused"], Value::Null, "{ack}");

    let stale = KubePods::new(&origin, None, None)
        .unwrap()
        .with_max_age(Duration::ZERO);
    api.deny_b.store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(pod_watch_once(&gateway, &stale).await, Some(Ok(3)));
    api.deny_b.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(pod_watch_once(&gateway, &stale).await.unwrap().is_err());
    let ack = gateway.ack().unwrap();
    assert!(
        ack["podsRefused"]
            .as_str()
            .is_some_and(|why| why.contains("slot 31")),
        "{ack}"
    );
}

#[tokio::test]
async fn a_bridge_the_gateway_cannot_list_refuses_no_canary() {
    let fx = Fixture::new();
    let gateway = fx.open();
    let (origin, _api) = serve_api().await;
    let mut policy = held_policy(2, 1000, 1, &fx.nfs_leaf(1000));
    policy["slots"][0]["bridge"] =
        json!({"namespace": "elsewhere", "podSelector": "app=checkpoint-server-other"});
    gateway.apply(&policy, Instant::now()).unwrap();
    let pods = KubePods::new(&origin, None, None).unwrap();
    assert!(matches!(
        pod_watch_once(&gateway, &pods).await,
        Some(Err(e)) if e.contains("bridge")
    ));
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["podsRefused"], Value::Null, "{ack}");
    assert_eq!(ack["bridgePeers"], json!(0), "{ack}");
    assert_eq!(
        pods.lookup("10.1.0.1".parse().unwrap()),
        Some((30, "uid-a".into()))
    );
}

#[tokio::test]
async fn a_slot_whose_listing_keeps_failing_is_forgotten() {
    let (origin, api) = serve_api().await;
    let pods = KubePods::new(&origin, None, None)
        .unwrap()
        .with_max_age(Duration::from_secs(60));
    let mut raw = held_policy(3, 1000, 1, "/nfs/x");
    raw["slots"]
        .as_array_mut()
        .unwrap()
        .push(json!({"slotId": 31, "namespace": "inference-b", "podSelector": "app=replies"}));
    let policy = Policy::from_json(&raw).unwrap();
    let t0 = Instant::now();
    assert_eq!(pods.refresh_at(&policy, t0).await, Ok(3));
    let at = |s: &str| pods.lookup(s.parse().unwrap());
    assert_eq!(at("10.1.0.20"), Some((31, "uid-f".into())));

    api.deny_b.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(
        pods.refresh_at(&policy, t0 + Duration::from_secs(10))
            .await
            .is_err()
    );
    assert_eq!(at("10.1.0.20"), Some((31, "uid-f".into())));
    assert!(
        pods.refresh_at(&policy, t0 + Duration::from_secs(61))
            .await
            .is_err()
    );
    assert_eq!(
        at("10.1.0.20"),
        None,
        "the slot's last good listing is trusted forever, so a reused address is that pod"
    );
    assert_eq!(at("10.1.0.1"), Some((30, "uid-a".into())));
}

#[tokio::test]
async fn the_ack_says_when_the_held_pin_or_nfs_move_failed() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1100);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1100));
    let rollback = serving_prefix(1200, "inc", 3);
    let gone = fx.nfs.join(format!("gone/elapsed_samples_1000/{RUN}"));
    let mut back = held_policy(7, 1200, 3, &gone.to_string_lossy());
    back["nfs"] = json!({"dir": fx.nfs.join("serving"), "offered": rollback});
    gateway.apply(&back, Instant::now()).unwrap();
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["version"], json!(7));
    assert_eq!(
        ack["heldPinned"],
        json!(false),
        "the ack reads as served: {ack}"
    );
    assert!(
        ack["heldError"].as_str().unwrap().contains(&real(1000)),
        "{ack}"
    );
    assert!(
        ack["nfsRefused"]
            .as_str()
            .unwrap()
            .contains("not a directory"),
        "{ack}"
    );

    write_checkpoint(&fx.root, 1000);
    let mut back = held_policy(8, 1200, 3, &fx.nfs_leaf(1000));
    back["nfs"] = json!({"dir": fx.nfs.join("serving"), "offered": rollback});
    gateway.apply(&back, Instant::now()).unwrap();
    let ack = gateway.ack().unwrap();
    assert_eq!(
        (&ack["heldPinned"], &ack["heldError"], &ack["nfsRefused"]),
        (&json!(true), &Value::Null, &Value::Null),
        "{ack}"
    );
}

#[test]
fn a_pin_never_races_eviction_into_a_partial_checkpoint() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    gateway.apply(&follow_inc(1), Instant::now()).unwrap();

    let guard = gateway.root_guard();
    let evicting = guard.lock();
    let dir = fx.root.join(real(1000));
    fs::remove_file(dir.join("dense_params/w")).unwrap();
    fs::remove_file(dir.join("emb_table/c/0/0")).unwrap();
    let g = gateway.clone();
    let leaf = fx.nfs_leaf(1000);
    let pin = std::thread::spawn(move || {
        g.apply(&held_policy(2, 1000, 1, &leaf), Instant::now())
            .unwrap()
            .held
    });
    std::thread::sleep(Duration::from_millis(300));
    fs::remove_dir_all(fx.root.join("elapsed_samples_000000000000001000")).unwrap();
    drop(evicting);
    let held = pin.join().unwrap();
    let pinned = scan_prefixes(&held_root_of(&fx.root));
    assert!(
        pinned.iter().all(|(_, files)| *files == file_sizes(1000)),
        "a partial checkpoint was pinned: {pinned:?}"
    );
    assert_eq!(held, None);
}

#[test]
fn a_partial_pin_is_pinned_again() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let partial = held_root_of(&fx.root).join(real(1000));
    fs::create_dir_all(&partial).unwrap();
    fs::write(
        partial.join("checksums.0.json"),
        file(1000, "checksums.0.json"),
    )
    .unwrap();
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    let applied = gateway
        .apply(&held_policy(2, 1000, 1, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    assert_eq!(applied.held, Some(real(1000)));
    assert_eq!(
        scan_prefixes(&held_root_of(&fx.root)),
        vec![(real(1000), file_sizes(1000))],
        "the partial pin is trusted"
    );
}

#[tokio::test]
async fn unpinning_a_held_checkpoint_drops_it_from_the_send_cache() {
    let fx = Fixture::new();
    let (gateway, _, _) = held_at_1000(&fx).await;
    let forgotten = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = forgotten.clone();
    gateway.on_unpin(move |prefix| seen.lock().push(prefix.to_string()));

    write_checkpoint(&fx.root, 1100);
    gateway
        .apply(&held_policy(3, 1200, 3, &fx.nfs_leaf(1100)), Instant::now())
        .unwrap();
    assert_eq!(
        *forgotten.lock(),
        vec![format!("{GATEWAY_DIR}/{HELD_DIR}/{}/", real(1000))],
        "the copy service keeps the unpinned checkpoint mapped"
    );
    gateway.apply(&follow_inc(4), Instant::now()).unwrap();
    assert_eq!(
        forgotten.lock().last(),
        Some(&format!("{GATEWAY_DIR}/{HELD_DIR}/{}/", real(1100)))
    );
}

#[tokio::test]
async fn replaced_canary_pods_free_their_admissions() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    let canary = canary_of_500(&fx, &gateway, 3, 600).await;
    let candidate = serving_prefix(1100, "cand", 2);
    let live = |uids: &[&str]| HashMap::from([(30, uids.iter().map(|u| u.to_string()).collect())]);
    gateway.note_live_pods(live(&["uid-1", "uid-2", "uid-3", "uid-4", "uid-5"]));
    for n in [1, 2] {
        assert_eq!(
            listed_prefixes(&mut svc, ip(n)).await,
            vec![candidate.clone()]
        );
    }
    let mut written_back = canary.clone();
    written_back["version"] = json!(4);
    written_back["slots"][0]["admitted"] = json!(["uid-1", "uid-2"]);
    gateway.apply(&written_back, Instant::now()).unwrap();

    gateway.note_live_pods(live(&["uid-3", "uid-6", "uid-7"]));
    let peers = StaticPeers(HashMap::from([
        (ip(3), (30, "uid-3".to_string())),
        (ip(6), (30, "uid-6".to_string())),
        (ip(7), (30, "uid-7".to_string())),
    ]));
    let mut replaced = GatewayService::new(
        FakeCopy {
            root: fx.root.clone(),
            fail_sends: false,
        },
        gateway.clone(),
        Arc::new(peers),
    );
    for (n, want) in [(6, &candidate), (7, &candidate), (3, &held)] {
        assert_eq!(
            &listed_prefixes(&mut replaced, ip(n)).await,
            &vec![want.clone()],
            "pod {n}: the canary's pods that are gone still fill it"
        );
    }
    let ack = gateway.ack().unwrap();
    let offered: Vec<&String> = ack["offered"].as_object().unwrap().keys().collect();
    assert_eq!(offered, ["uid-3", "uid-6", "uid-7"], "{ack}");
}

#[tokio::test]
async fn replaced_pods_free_their_share_of_the_rolling_budget() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    let mut rolling = canary_of_500(&fx, &gateway, 3, 600).await;
    rolling["version"] = json!(4);
    rolling["mode"] = json!("rolling");
    rolling["slots"][0]["released"] = json!(true);
    rolling["slots"][0]["budget"] = json!({"pods": 2, "perMinutes": 60});
    gateway.apply(&rolling, Instant::now()).unwrap();
    let candidate = serving_prefix(1100, "cand", 2);
    let live = |uids: &[&str]| HashMap::from([(30, uids.iter().map(|u| u.to_string()).collect())]);
    gateway.note_live_pods(live(&["uid-1", "uid-2", "uid-3"]));
    for (n, want) in [(1, &candidate), (2, &candidate), (3, &held)] {
        assert_eq!(
            &listed_prefixes(&mut svc, ip(n)).await,
            &vec![want.clone()],
            "pod {n}"
        );
    }

    gateway.note_live_pods(live(&["uid-3", "uid-6"]));
    let peers = StaticPeers(HashMap::from([
        (ip(3), (30, "uid-3".to_string())),
        (ip(6), (30, "uid-6".to_string())),
    ]));
    let mut replaced = GatewayService::new(
        FakeCopy {
            root: fx.root.clone(),
            fail_sends: false,
        },
        gateway.clone(),
        Arc::new(peers),
    );
    for n in [6, 3] {
        assert_eq!(
            listed_prefixes(&mut replaced, ip(n)).await,
            vec![candidate.clone()],
            "pod {n}: the rolling budget still counts the pods that are gone"
        );
    }
}

#[test]
fn a_seed_never_replaces_a_pin_made_while_it_copied() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    gateway.apply(&follow_inc(1), Instant::now()).unwrap();
    gateway
        .apply(&held_policy(2, 1000, 1, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    gateway.reconcile_durable().unwrap();
    fs::remove_dir_all(
        gateway
            .held_root()
            .join("elapsed_samples_000000000000001000"),
    )
    .unwrap();
    gateway.state.lock().held = None;

    let root_guard = gateway.root_guard();
    let blocked = root_guard.lock();
    let pinning = {
        let g = gateway.clone();
        std::thread::spawn(move || g.pin_local(&real(1000)))
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while !gateway.pins.is_locked() {
        assert!(
            Instant::now() < deadline,
            "the pin never took the pins lock"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let seeding = {
        let g = gateway.clone();
        std::thread::spawn(move || g.reconcile_durable())
    };
    let staging = gateway
        .held_root()
        .join(format!(".elapsed_samples_000000000000001000.{RUN}.seed"));
    let want = tree_files(&fx.root.join(real(1000))).unwrap();
    while tree_files(&staging).ok().as_ref() != Some(&want) {
        assert!(
            Instant::now() < deadline,
            "the seed never copied the durable pin"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(blocked);
    pinning.join().unwrap().unwrap();
    let _ = seeding.join().unwrap();

    use std::os::unix::fs::MetadataExt;
    let ino = |root: &Path| {
        fs::metadata(root.join(real(1000)).join("dense_params/w"))
            .unwrap()
            .ino()
    };
    assert_eq!(
        ino(&gateway.held_root()),
        ino(&fx.root),
        "the seed's copy replaced the pin linked from the serving root"
    );
    assert_eq!(gateway.ack().unwrap()["heldPinned"], json!(true));
}

#[test]
fn a_policy_of_another_binding_is_applied_not_compared() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    let leaf = fx.nfs_leaf(1000);
    let mut old = held_policy(40, 1000, 1, &leaf);
    old["binding"] = json!(7);
    gateway.apply(&old, Instant::now()).unwrap();
    gateway.reconcile_durable().unwrap();

    let mut new = held_policy(1, 1100, 2, &leaf);
    new["binding"] = json!(8);
    gateway
        .apply(&new, Instant::now())
        .expect("binding 8's policy was compared with binding 7's version");
    assert_eq!(gateway.ack().unwrap()["version"], json!(1));

    let fresh_root = tempfile::tempdir().unwrap();
    let redeployed = Gateway::open(GatewayConfig {
        binding: Some("9".into()),
        ..fx.config(fresh_root.path(), 0)
    });
    assert_eq!(redeployed.policy(), None);
    let err = redeployed.apply(&old, Instant::now()).unwrap_err();
    assert!(
        err.contains("binding 7") && err.contains("binding 9"),
        "{err}"
    );
}

#[tokio::test]
async fn a_planted_state_file_never_reaches_outside_the_serving_root() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    gateway.apply(&follow_inc(1), Instant::now()).unwrap();
    let mut svc = fx.service(&gateway, &fx.root);
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving_prefix(1000, "inc", 1)]
    );
    drop(svc);
    drop(gateway);

    let outside = fx.nfs.file_name().unwrap().to_str().unwrap().to_string();
    fs::write(fx.nfs.join("secret"), b"not a checkpoint").unwrap();
    let escape = format!("elapsed_samples_000000000000001000/../../{outside}");
    let planted = "elapsed_samples_000000000000009000/x-g1";
    let file = fx.root.join(GATEWAY_DIR).join(STATE_FILE);
    let mut saved: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    saved["latest"] = json!(escape);
    saved["ledger"]["names"][planted] = json!(escape);
    saved["ledger"]["peers"]["pod:uid-1"] = json!({
        "slotId": 30,
        "uid": "uid-1",
        "offers": [{"serving": planted, "seq": 9000, "source": {"upstream": escape}}],
    });
    fs::write(&file, saved.to_string()).unwrap();

    let reopened = fx.open();
    let mut svc = fx.service(&reopened, &fx.root);
    let got = send(&mut svc, ip(1), &format!("{planted}/secret"), 16).await;
    assert_eq!(
        got.map_err(|e| e.code()),
        Err(tonic::Code::NotFound),
        "a planted source read a file outside the serving root"
    );
    assert_eq!(reopened.latest(), None);
    assert_eq!(
        reopened.state.lock().ledger.real_of(planted),
        None,
        "a planted name was loaded"
    );
}

#[tokio::test]
async fn a_saved_state_never_outranks_the_control_planes_policy() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    let leaf = fx.nfs_leaf(1000);
    gateway
        .apply(&held_policy(2, 1000, 1, &leaf), Instant::now())
        .unwrap();
    drop(gateway);

    let file = fx.root.join(GATEWAY_DIR).join(STATE_FILE);
    let mut saved: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    saved["policy"]["version"] = json!(u64::MAX);
    saved["policy"]["mode"] = json!("follow");
    fs::write(&file, saved.to_string()).unwrap();

    let reopened = fx.open();
    reopened
        .apply(&held_policy(3, 1000, 1, &leaf), Instant::now())
        .expect("the saved state's version outranked the control plane's policy");
    assert_eq!(reopened.ack().unwrap()["version"], json!(3));
    let mut svc = fx.service(&reopened, &fx.root);
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving_prefix(1000, "inc", 1)]
    );
    let err = reopened
        .apply(&held_policy(2, 1000, 1, &leaf), Instant::now())
        .unwrap_err();
    assert!(err.contains("older"), "{err}");
}

#[test]
fn a_saved_state_without_a_binding_is_not_a_bound_gateways() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 1000);
    let gateway = fx.open();
    gateway.note_mirrored(&real(1000));
    let mut unbound = held_policy(40, 1000, 1, &fx.nfs_leaf(1000));
    unbound.as_object_mut().unwrap().remove("binding");
    gateway.apply(&unbound, Instant::now()).unwrap();
    gateway.reconcile_durable().unwrap();

    let fresh_root = tempfile::tempdir().unwrap();
    let bound = Gateway::open(GatewayConfig {
        binding: Some("9".into()),
        ..fx.config(fresh_root.path(), 0)
    });
    assert_eq!(
        bound.policy(),
        None,
        "a state that names no binding was loaded by binding 9's gateway"
    );
}

#[tokio::test]
async fn a_send_that_never_finished_is_not_reported_downloaded() {
    let fx = Fixture::new();
    write_checkpoint(&fx.root, 500);
    let gateway = fx.open();
    gateway.note_mirrored(&real(500));
    let follow = json!({"version": 1, "mode": "follow"});
    gateway.apply(&follow, Instant::now()).unwrap();
    let mut svc = fx.service(&gateway, &fx.root);
    let name = format!("{}/dense_params/w", real(500));
    for n in [1, 2, 3] {
        assert_eq!(listed_prefixes(&mut svc, ip(n)).await, vec![real(500)]);
    }
    let send_request = |rdma: bool| {
        let (all, p, s) = encode_names(vec![name.as_bytes().to_vec()]);
        let mut buf = get_bytes_mut();
        buf.put_string(1, &all);
        buf.put_repeated_ints([2, 3, 4, 5], [&p, &s, &[0], &[96]]);
        if rdma {
            buf.put_string(6, b"endpoint");
        }
        body::Body::new(freeze(buf))
    };

    let response = svc
        .call(request(SEND, ip(1), send_request(false)))
        .await
        .unwrap();
    drop(response);
    let mut broken = GatewayService::new(
        FakeCopy {
            root: fx.root.clone(),
            fail_sends: true,
        },
        gateway.clone(),
        fx.peers.clone(),
    );
    let response = broken
        .call(request(SEND, ip(2), send_request(false)))
        .await
        .unwrap();
    let _ = http_body_util::BodyExt::collect(response.into_body()).await;
    let response = svc
        .call(request(SEND, ip(3), send_request(true)))
        .await
        .unwrap();
    let _ = http_body_util::BodyExt::collect(response.into_body()).await;

    let ack = gateway.ack().unwrap();
    let downloaded = &ack["downloaded"];
    for uid in ["uid-1", "uid-2"] {
        assert_eq!(
            downloaded[uid]["ok"],
            json!(false),
            "{uid}: an unfinished transfer reads as downloaded: {ack}"
        );
    }
    assert_eq!(downloaded["uid-1"]["bytes"], json!(0), "{ack}");
    assert_eq!(downloaded["uid-3"]["bytes"], json!(0), "{ack}");
    assert_eq!(
        downloaded["uid-3"]["rdmaRequestedBytes"],
        json!(96),
        "{ack}"
    );
}

#[test]
fn concurrent_state_writes_never_mix() {
    let dir = tempfile::tempdir().unwrap();
    let bodies = [vec![b'a'; 8 << 20], vec![b'b'; (8 << 20) - 4096]];
    let whole = |bytes: &[u8]| bodies.iter().any(|b| b.as_slice() == bytes);
    let done = std::sync::atomic::AtomicBool::new(false);
    let (failed, mixed) = std::thread::scope(|scope| {
        let reader = scope.spawn(|| {
            let mut mixed = 0;
            while !done.load(std::sync::atomic::Ordering::SeqCst) {
                if let Ok(bytes) = fs::read(dir.path().join(STATE_FILE))
                    && !whole(&bytes)
                {
                    mixed += 1;
                }
            }
            mixed
        });
        let writers: Vec<_> = bodies
            .iter()
            .map(|body| {
                scope.spawn(|| {
                    (0..20)
                        .filter(|_| write_atomic(dir.path(), STATE_FILE, body).is_err())
                        .count()
                })
            })
            .collect();
        let failed: usize = writers.into_iter().map(|w| w.join().unwrap()).sum();
        done.store(true, std::sync::atomic::Ordering::SeqCst);
        (failed, reader.join().unwrap())
    });
    assert_eq!(
        (failed, mixed),
        (0, 0),
        "two saves shared one temp file: writes failed, or state.json mixed both"
    );
    assert!(whole(&fs::read(dir.path().join(STATE_FILE)).unwrap()));
}

#[tokio::test]
async fn a_planted_link_in_the_state_dir_is_never_served() {
    let fx = Fixture::new();
    let outside = tempfile::tempdir().unwrap();
    let secret = outside.path().join("token");
    let canary = b"CANARY-not-a-checkpoint-0193";
    fs::write(&secret, canary).unwrap();
    let planted = fx.state.join("s0").join(HELD_DIR).join(real(1000));
    for (name, data) in files(1000) {
        let path = planted.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, data).unwrap();
    }
    fs::remove_file(planted.join("dense_params/w")).unwrap();
    std::os::unix::fs::symlink(&secret, planted.join("dense_params/w")).unwrap();
    let victim = outside.path().join("victim");
    fs::write(&victim, b"untouched").unwrap();
    for dir in [fx.root.join(GATEWAY_DIR), fx.state.join("s0")] {
        fs::create_dir_all(&dir).unwrap();
        std::os::unix::fs::symlink(&victim, dir.join(format!(".{STATE_FILE}.tmp"))).unwrap();
    }

    let gateway = fx.open();
    gateway
        .apply(&held_policy(2, 1000, 1, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    let _ = gateway.reconcile_durable();
    let mut served = Vec::new();
    walk_files(&held_root_of(&fx.root), "", &mut served);
    for (name, _) in &served {
        let path = held_root_of(&fx.root).join(name);
        assert_ne!(
            fs::read(&path).ok().as_deref(),
            Some(&canary[..]),
            "{name} serves the linked secret"
        );
    }
    let mut svc = fx.service(&gateway, &fx.root);
    let held = serving_prefix(1000, "inc", 1);
    if let Ok(bytes) = send(
        &mut svc,
        ip(1),
        &format!("{held}/dense_params/w"),
        canary.len(),
    )
    .await
    {
        assert_ne!(bytes, canary, "the copy port serves the linked secret");
    }
    assert_eq!(
        fs::read(&victim).unwrap(),
        b"untouched",
        "a planted temp link was written through"
    );
}

#[test]
fn policy_tls_needs_an_https_url_and_its_ca() {
    #[derive(Parser)]
    struct Server {
        #[command(flatten)]
        gateway: GatewayArgs,
    }
    let parse = |args: &[&str]| {
        Server::try_parse_from(std::iter::once("checkpoint-server").chain(args.iter().copied()))
    };
    let https = "https://cp:8443/api/prod-trainers/7/serving-policy";
    let cert = ["--policy-tls-cert", "/c.pem", "--policy-tls-key", "/k.pem"];
    let args: Vec<&str> = ["--policy-url", https].into_iter().chain(cert).collect();
    assert!(
        parse(&args).is_err(),
        "a client certificate without --policy-tls-ca is dropped"
    );

    let err = parse(&["--policy-url", https])
        .unwrap()
        .gateway
        .tls()
        .unwrap_err();
    assert!(err.contains("--policy-tls-ca"), "{err}");

    let http = "http://cp:8080/api/prod-trainers/7/serving-policy";
    let err = parse(&["--policy-url", http, "--policy-tls-ca", "/ca.pem"])
        .unwrap()
        .gateway
        .tls()
        .unwrap_err();
    assert!(err.contains("https"), "{err}");
    assert!(matches!(
        parse(&["--policy-url", http]).unwrap().gateway.tls(),
        Ok(None)
    ));
}

#[derive(Default)]
struct Directory(parking_lot::RwLock<HashMap<IpAddr, (i64, String)>>);

impl PeerDirectory for Directory {
    fn lookup(&self, ip: IpAddr) -> Option<(i64, String)> {
        self.0.read().get(&ip).cloned()
    }
}

async fn download_all(
    svc: &mut GatewayService<FakeCopy>,
    from: IpAddr,
    prefix: &str,
    elapsed: u64,
) {
    for (name, data) in files(elapsed) {
        let got = send(svc, from, &format!("{prefix}/{name}"), data.len()).await;
        assert_eq!(got.unwrap(), data, "{prefix}/{name}");
    }
}

#[tokio::test]
async fn a_pod_that_fetched_before_the_directory_listed_it_is_named_on_the_next_listing() {
    let fx = Fixture::new();
    let (gateway, _, held) = held_at_1000(&fx).await;
    let directory = Arc::new(Directory::default());
    let mut svc = GatewayService::new(
        FakeCopy {
            root: fx.root.clone(),
            fail_sends: false,
        },
        gateway.clone(),
        directory.clone(),
    );
    assert_eq!(listed_prefixes(&mut svc, ip(7)).await, vec![held.clone()]);
    download_all(&mut svc, ip(7), &held, 1000).await;
    assert_eq!(gateway.ack().unwrap()["unknownPeers"], json!(1));

    directory.0.write().insert(ip(7), (30, "uid-7".to_string()));
    gateway.identify_peers(directory.as_ref());
    let ack = gateway.ack().unwrap();
    assert_eq!(ack["unknownPeers"], json!(0), "{ack}");
    assert_eq!(ack["offered"]["uid-7"], json!(held), "{ack}");
    assert_eq!(ack["downloaded"]["uid-7"]["prefix"], json!(held), "{ack}");
    assert_eq!(ack["downloaded"]["uid-7"]["ok"], json!(true), "{ack}");

    gateway.reconcile_durable().unwrap();
    drop(svc);
    drop(gateway);
    let restarted = fx.open();
    let ack = restarted.ack().unwrap();
    assert_eq!(ack["unknownPeers"], json!(0), "{ack}");
    assert_eq!(ack["downloaded"]["uid-7"]["ok"], json!(true), "{ack}");
}

#[tokio::test]
async fn an_unknown_peer_that_stopped_asking_is_forgotten_after_its_bound() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    assert_eq!(listed_prefixes(&mut svc, ip(99)).await, vec![held.clone()]);
    gateway
        .apply(&held_policy(2, 1000, 1, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    drop(svc);
    drop(gateway);

    let file = fx.root.join(GATEWAY_DIR).join(STATE_FILE);
    let mut saved: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let gone = json!({
        "slotId": null,
        "uid": null,
        "offers": [{"serving": held, "seq": 1000, "source": {"held": real(1000)}}],
    });
    let mut long_ago = gone.clone();
    long_ago["seen"] = json!(1);
    saved["ledger"]["peers"]["ip:10.0.0.98"] = gone;
    saved["ledger"]["peers"]["ip:10.0.0.97"] = long_ago;
    fs::write(&file, saved.to_string()).unwrap();

    let reopened = fx.open();
    let ack = reopened.ack().unwrap();
    assert_eq!(
        ack["unknownPeers"],
        json!(1),
        "only 10.0.0.99, which asked just now, is still unknown: {ack}"
    );
    let saved: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let peers = saved["ledger"]["peers"].as_object().unwrap();
    assert!(
        !peers.contains_key("ip:10.0.0.98") && !peers.contains_key("ip:10.0.0.97"),
        "{saved}"
    );
}

#[tokio::test]
async fn a_restarted_gateway_still_reports_what_each_pod_downloaded() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);
    download_all(&mut svc, ip(1), &held, 1000).await;
    let before = gateway.ack().unwrap()["downloaded"].clone();
    assert_eq!(before["uid-1"]["ok"], json!(true), "{before}");
    gateway
        .apply(&held_policy(2, 1000, 1, &fx.nfs_leaf(1000)), Instant::now())
        .unwrap();
    drop(svc);
    drop(gateway);

    let restarted = fx.open();
    let ack = restarted.ack().unwrap();
    assert_eq!(ack["downloaded"], before, "{ack}");
    assert_eq!(ack["podsSeqMax"], json!(1000), "{ack}");
}

#[tokio::test]
async fn a_rolled_back_candidates_checkpoint_is_never_served_as_the_restored_runs() {
    let fx = Fixture::new();
    let (gateway, mut svc, _) = held_at_1000(&fx).await;
    canary_of_500(&fx, &gateway, 3, 600).await;
    let leaf = fx.nfs_leaf(1000);
    let held = serving_prefix(2000, INC, 3);
    gateway
        .apply(&held_policy(4, 2000, 3, &leaf), Instant::now())
        .unwrap();
    let mut follow = held_policy(5, 2000, 3, &leaf);
    follow["mode"] = json!("follow");
    follow["released"] =
        json!({"run": INC, "generation": 3, "seqOffset": 1000, "from": real_of(1100, INC)});
    gateway.apply(&follow, Instant::now()).unwrap();

    write_checkpoint_of(&fx.root, 1300, CAND);
    gateway.note_mirrored(&real_of(1300, CAND));
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![held.clone()],
        "the candidate's 1300 counted as the restored trainer's first checkpoint"
    );

    assert!(gateway.rewind_to(&real_of(1100, INC)));
    write_checkpoint_of(&fx.root, 1100, INC);
    gateway.note_mirrored(&real_of(1100, INC));
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving_prefix(2100, INC, 3)]
    );
    write_checkpoint_of(&fx.root, 1400, CAND);
    gateway.note_mirrored(&real_of(1400, CAND));
    let disguised = serving_prefix(2400, INC, 3);
    for pod in [ip(1), ip(4)] {
        let got = listed_prefixes(&mut svc, pod).await;
        assert!(
            !got.contains(&disguised),
            "pod {pod} was offered the candidate's 1400 as {got:?}"
        );
        let err = send(&mut svc, pod, &format!("{disguised}/dense_params/w"), 96)
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound, "pod {pod}");
    }
}

#[tokio::test]
async fn the_serving_dir_moves_to_the_released_candidate_once_every_slot_is_released() {
    let fx = Fixture::new();
    let (gateway, _, _) = held_at_1000(&fx).await;
    let mut released = canary_of_500(&fx, &gateway, 3, 600).await;
    let dir = fx.nfs.join("serving");
    released["version"] = json!(4);
    released["mode"] = json!("rolling");
    released["slots"][0]["released"] = json!(true);
    let candidate = serving_prefix(1100, CAND, 2);
    for wrong in [fx.nfs_leaf_of(400, CAND), fx.nfs_leaf_of(500, INC)] {
        released["nfs"] = json!({"dir": dir, "offered": candidate, "offeredPath": wrong});
        let refused = gateway
            .apply(&released, Instant::now())
            .unwrap()
            .nfs_refused
            .unwrap();
        assert!(refused.contains(&wrong), "{refused}");
        assert_eq!(gateway.ack().unwrap()["nfsRefused"], json!(refused));
        assert_eq!(serving_dir_entries(&dir), Vec::<String>::new());
    }

    let leaf = fx.nfs_leaf_of(500, CAND);
    released["nfs"] = json!({"dir": dir, "offered": candidate, "offeredPath": leaf});
    let applied = gateway.apply(&released, Instant::now()).unwrap();
    assert_eq!(applied.nfs_refused, None);
    assert_eq!(serving_dir_entries(&dir), vec![candidate.clone()]);
    assert_eq!(
        fs::read_link(dir.join(&candidate).join("completed")).unwrap(),
        Path::new(&leaf).join("completed")
    );
    assert_eq!(gateway.ack().unwrap()["nfsRefused"], Value::Null);
}

#[tokio::test]
async fn a_restored_trainer_that_restarted_is_still_followed() {
    let fx = Fixture::new();
    let (gateway, mut svc, _) = held_at_1000(&fx).await;
    let leaf = fx.nfs_leaf(1000);
    let held = serving_prefix(2000, INC, 3);
    gateway
        .apply(&held_policy(3, 2000, 3, &leaf), Instant::now())
        .unwrap();
    let mut follow = held_policy(4, 2000, 3, &leaf);
    follow["mode"] = json!("follow");
    follow["released"] =
        json!({"run": "inc-0", "generation": 3, "seqOffset": 1000, "from": real_of(1100, "inc-0")});
    gateway.apply(&follow, Instant::now()).unwrap();
    assert_eq!(listed_prefixes(&mut svc, ip(1)).await, vec![held.clone()]);

    for (elapsed, run) in [(1200, "inc-1"), (1300, "inc-2")] {
        write_checkpoint_of(&fx.root, elapsed, run);
        gateway.note_mirrored(&real_of(elapsed, run));
        let next = serving_prefix(elapsed + 1000, "inc-0", 3);
        for pod in [ip(1), ip(4)] {
            assert_eq!(
                listed_prefixes(&mut svc, pod).await,
                vec![next.clone()],
                "pod {pod}: the restored trainer's {run} left serving on the restore point"
            );
        }
        assert_eq!(
            send(&mut svc, ip(1), &format!("{next}/dense_params/w"), 96)
                .await
                .unwrap(),
            file(elapsed, "dense_params/w")
        );
    }

    write_checkpoint_of(&fx.root, 1400, "cand-0");
    gateway.note_mirrored(&real_of(1400, "cand-0"));
    let disguised = serving_prefix(2400, "inc-0", 3);
    for pod in [ip(1), ip(5)] {
        let got = listed_prefixes(&mut svc, pod).await;
        assert!(
            !got.contains(&disguised),
            "pod {pod} was offered the candidate's 1400 as {got:?}"
        );
    }
}

#[tokio::test]
async fn a_candidate_that_restarted_is_still_released() {
    let fx = Fixture::new();
    let (gateway, mut svc, held) = held_at_1000(&fx).await;
    assert!(gateway.rewind_to(&real_of(500, "cand-0")));
    write_checkpoint_of(&fx.root, 500, "cand-0");
    gateway.note_mirrored(&real_of(500, "cand-0"));
    let mut canary = held_policy(3, 1000, 1, &fx.nfs_leaf(1000));
    canary["mode"] = json!("canary");
    canary["released"] = json!({"run": "cand-0", "generation": 2, "seqOffset": 600});
    canary["slots"][0]["canaryPods"] = json!(1);
    gateway.apply(&canary, Instant::now()).unwrap();
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![serving_prefix(1100, "cand-0", 2)]
    );

    write_checkpoint_of(&fx.root, 600, "cand-1");
    gateway.note_mirrored(&real_of(600, "cand-1"));
    let restarted = serving_prefix(1200, "cand-0", 2);
    assert_eq!(
        listed_prefixes(&mut svc, ip(1)).await,
        vec![restarted.clone()],
        "the canary pod was left on the held checkpoint after the candidate restarted"
    );
    assert_eq!(
        send(&mut svc, ip(1), &format!("{restarted}/dense_params/w"), 96)
            .await
            .unwrap(),
        file(600, "dense_params/w")
    );
    assert_eq!(listed_prefixes(&mut svc, ip(2)).await, vec![held.clone()]);

    let mut rolling = canary.clone();
    rolling["version"] = json!(4);
    rolling["mode"] = json!("rolling");
    rolling["slots"][0]["released"] = json!(true);
    gateway.apply(&rolling, Instant::now()).unwrap();
    for pod in [ip(2), ip(3)] {
        assert_eq!(
            listed_prefixes(&mut svc, pod).await,
            vec![restarted.clone()],
            "pod {pod} of the released slot was left on the held checkpoint"
        );
    }
}

#[tokio::test]
async fn the_serving_dir_moves_to_a_restarted_candidates_checkpoint() {
    let fx = Fixture::new();
    let (gateway, _, _) = held_at_1000(&fx).await;
    assert!(gateway.rewind_to(&real_of(600, "cand-1")));
    write_checkpoint_of(&fx.root, 600, "cand-1");
    gateway.note_mirrored(&real_of(600, "cand-1"));
    let dir = fx.nfs.join("serving");
    let candidate = serving_prefix(1200, "cand-0", 2);
    let mut released = held_policy(3, 1000, 1, &fx.nfs_leaf(1000));
    released["mode"] = json!("rolling");
    released["released"] = json!({"run": "cand-0", "generation": 2, "seqOffset": 600});
    released["slots"][0]["released"] = json!(true);
    let other = fx.nfs_leaf_of(600, "inc-1");
    released["nfs"] = json!({"dir": dir, "offered": candidate, "offeredPath": other});
    let refused = gateway
        .apply(&released, Instant::now())
        .unwrap()
        .nfs_refused
        .unwrap();
    assert!(refused.contains(&other), "{refused}");
    assert_eq!(serving_dir_entries(&dir), Vec::<String>::new());

    let leaf = fx.nfs_leaf_of(600, "cand-1");
    released["version"] = json!(4);
    released["nfs"] = json!({"dir": dir, "offered": candidate, "offeredPath": leaf});
    let applied = gateway.apply(&released, Instant::now()).unwrap();
    assert_eq!(
        applied.nfs_refused, None,
        "the restarted candidate's checkpoint was refused"
    );
    assert_eq!(serving_dir_entries(&dir), vec![candidate.clone()]);
    assert_eq!(
        fs::read_link(dir.join(&candidate).join("completed")).unwrap(),
        Path::new(&leaf).join("completed")
    );
}

#[tokio::test]
async fn a_send_in_flight_when_the_listing_names_its_pod_counts_as_its_download() {
    let fx = Fixture::new();
    let (gateway, _, held) = held_at_1000(&fx).await;
    let directory = Arc::new(Directory::default());
    let mut svc = GatewayService::new(
        FakeCopy {
            root: fx.root.clone(),
            fail_sends: false,
        },
        gateway.clone(),
        directory.clone(),
    );
    assert_eq!(listed_prefixes(&mut svc, ip(7)).await, vec![held.clone()]);
    let name = format!("{held}/dense_params/w");
    let (all, p, s) = encode_names(vec![name.as_bytes().to_vec()]);
    let mut buf = get_bytes_mut();
    buf.put_string(1, &all);
    buf.put_repeated_ints([2, 3, 4, 5], [&p, &s, &[0], &[96]]);
    let in_flight = svc
        .call(request(SEND, ip(7), body::Body::new(freeze(buf))))
        .await
        .unwrap();

    directory.0.write().insert(ip(7), (30, "uid-7".to_string()));
    gateway.identify_peers(directory.as_ref());
    let _ = http_body_util::BodyExt::collect(in_flight.into_body()).await;
    let ack = gateway.ack().unwrap();
    assert_eq!(
        ack["downloaded"]["uid-7"]["prefix"],
        json!(held),
        "the Send in flight when the pod was named was not counted as its download: {ack}"
    );
    assert_eq!(ack["downloaded"]["uid-7"]["bytes"], json!(96), "{ack}");
    assert_eq!(ack["downloaded"]["uid-7"]["ok"], json!(true), "{ack}");
    assert_eq!(ack["unknownPeers"], json!(0), "{ack}");
}
