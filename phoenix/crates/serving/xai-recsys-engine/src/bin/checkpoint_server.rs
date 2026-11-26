// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use clap::Parser;
#[cfg(target_os = "linux")]
use log::{error, info, warn};
#[cfg(target_os = "linux")]
use std::{error::Error, time::Duration};
#[cfg(target_os = "linux")]
use tokio::signal;

#[cfg(target_os = "linux")]
use {
    axum::{Router, routing::get},
    prometheus::TextEncoder,
    std::net::SocketAddr,
    std::sync::Arc,
    tokio::net::{TcpListener, TcpSocket},
    tonic::transport::{Server, server::TcpIncoming},
    xai_recsys_engine::{
        checkpoint_gateway::{
            self, Gateway, GatewayArgs, GatewayConfig, GatewayService, KubePods, NoPeers,
            PeerDirectory, PolicyClient,
        },
        checkpoint_proxy::CheckpointProxy,
        r#const::ROOT_DIR,
        copy::{CopyService, Sender},
    },
};

#[derive(Parser)]
#[command(about = "Cross-DC checkpoint proxy for recsys inference")]
struct Args {
    #[clap(long)]
    copy_url: String,

    #[clap(long)]
    server_index: Option<usize>,

    #[clap(long)]
    num_servers: usize,

    #[clap(long)]
    num_trainers: Option<usize>,

    #[clap(long, default_value_t = 9988)]
    port: u16,

    #[clap(long, default_value_t = 9090)]
    metrics_port: u16,

    #[clap(long, default_value_t = 30)]
    poll_interval: u64,

    #[clap(long, default_value_t = 2)]
    max_entries: usize,

    #[clap(long, default_value_t = true)]
    verify_checksums: bool,

    #[clap(long, default_value_t = 8)]
    download_concurrency: usize,

    #[clap(long, default_value_t = 600)]
    request_timeout: u64,

    #[clap(long, requires = "tls_key")]
    tls_cert: Option<String>,
    #[clap(long, requires = "tls_cert")]
    tls_key: Option<String>,
    #[clap(long, requires = "tls_cert")]
    tls_client_ca: Option<String>,

    #[clap(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=63))]
    dscp: u8,

    #[cfg(target_os = "linux")]
    #[command(flatten)]
    gateway: GatewayArgs,
}

#[cfg(target_os = "linux")]
fn derive_server_index() -> Option<usize> {
    let hostname = std::env::var("HOSTNAME").ok()?;
    let idx = hostname.rfind('-')?;
    hostname[idx + 1..].parse().ok()
}

#[cfg(target_os = "linux")]
fn bind_grpc_listener(addr: SocketAddr, dscp: u8) -> std::io::Result<TcpListener> {
    let socket = TcpSocket::new_v4()?;
    socket.set_reuseaddr(true)?;
    socket.set_tos_v4(u32::from(dscp) << 2)?;
    socket.bind(addr)?;
    socket.listen(1024)
}

#[cfg(target_os = "linux")]
async fn metrics_handler() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder
        .encode_to_string(&metric_families)
        .unwrap_or_default()
}

#[cfg(target_os = "linux")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = Args::parse();

    let server_index = match args.server_index {
        Some(idx) => idx,
        None => derive_server_index().ok_or(
            "cannot derive server index: set --server-index or ensure HOSTNAME ends with -N",
        )?,
    };

    if server_index >= args.num_servers {
        return Err(format!(
            "server_index {} >= num_servers {}",
            server_index, args.num_servers,
        )
        .into());
    }

    info!(
        "checkpoint-server starting: copy_url={}, index={}/{}, port={}, metrics_port={}, poll={}s, max_entries={}, request_timeout={}s, dscp={}",
        args.copy_url,
        server_index,
        args.num_servers,
        args.port,
        args.metrics_port,
        args.poll_interval,
        args.max_entries,
        args.request_timeout,
        args.dscp,
    );

    let gateway = if let Some(policy_url) = &args.gateway.policy_url {
        info!("prod checkpoint gateway: serving policy from {policy_url}");
        let config = GatewayConfig {
            root: ROOT_DIR.into(),
            state_dir: args.gateway.gateway_state_dir.as_ref().map(Into::into),
            server_index,
            policy_fresh: Duration::from_secs(args.gateway.policy_fresh_secs),
            name: std::env::var("HOSTNAME").unwrap_or_default(),
            binding: args.gateway.binding(),
            unknown_peer_ttl: Duration::from_secs(args.gateway.unknown_peer_secs),
        };
        let client = PolicyClient::new(policy_url, args.gateway.tls()?.as_ref())?;
        let pods = match KubePods::in_cluster() {
            Ok(pods) => Some(Arc::new(
                pods.with_max_age(Duration::from_secs(args.gateway.pod_stale_secs)),
            )),
            Err(e) => {
                warn!("prod checkpoint gateway: no pod directory, every peer is unknown: {e}");
                None
            }
        };
        Some((Arc::new(Gateway::open(config)), client, pods))
    } else {
        None
    };
    let copy_url = args.copy_url.clone();
    let evict_guard = gateway.as_ref().map(|(gateway, _, _)| gateway.root_guard());
    let make_proxy = move || {
        let proxy = CheckpointProxy::new(
            copy_url.clone(),
            server_index,
            args.num_servers,
            args.num_trainers,
            args.max_entries,
            args.verify_checksums,
            args.download_concurrency,
            Duration::from_secs(args.request_timeout),
        );
        match &evict_guard {
            Some(guard) => proxy.with_evict_guard(guard.clone()),
            None => proxy,
        }
    };
    let proxy = make_proxy();

    let grpc_addr = format!("0.0.0.0:{}", args.port).parse()?;
    let grpc_listener = bind_grpc_listener(grpc_addr, args.dscp)?;
    info!("CopyService listening on {}", grpc_addr);

    let mut server = Server::builder();
    if let Some(tls) = xai_recsys_engine::tls::ServerTlsOptions::from_paths(
        args.tls_cert,
        args.tls_key,
        args.tls_client_ca,
    )? {
        server = server.tls_config(tls.load()?)?;
    }
    let sender = Arc::new(Sender::new(args.max_entries));
    let copy_service = CopyService::from_arc(sender.clone());
    let router = match &gateway {
        Some((gateway, _, pods)) => {
            gateway.on_unpin(move |prefix| sender.forget(prefix));
            let peers: Arc<dyn PeerDirectory> = match pods {
                Some(pods) => pods.clone(),
                None => Arc::new(NoPeers),
            };
            server.add_service(GatewayService::new(copy_service, gateway.clone(), peers))
        }
        None => server.add_service(copy_service),
    };
    let server_handle = tokio::spawn(async move {
        router
            .serve_with_incoming(TcpIncoming::from(grpc_listener).with_nodelay(Some(true)))
            .await
    });

    let metrics_addr: std::net::SocketAddr = format!("0.0.0.0:{}", args.metrics_port).parse()?;
    let metrics_app = Router::new().route("/metrics", get(metrics_handler));
    info!(
        "Metrics endpoint listening on http://{}/metrics",
        metrics_addr
    );

    let metrics_listener = tokio::net::TcpListener::bind(metrics_addr).await?;
    let metrics_handle =
        tokio::spawn(async move { axum::serve(metrics_listener, metrics_app).await });

    xai_recsys_engine::checkpoint_proxy::cleanup_stale_staging_dirs();

    if proxy.has_existing_checkpoints() {
        info!("found pre-existing checkpoints in /dev/shm, ready to serve");
    }

    if let Some((gateway, client, pods)) = gateway {
        if let Some(pods) = pods {
            tokio::spawn(checkpoint_gateway::run_pod_watch(
                gateway.clone(),
                pods,
                Duration::from_secs(args.gateway.pod_refresh_secs.max(1)),
            ));
        }
        let every = Duration::from_secs(args.gateway.policy_poll_secs.max(1));
        tokio::spawn(checkpoint_gateway::run_policy_loop(
            gateway.clone(),
            client,
            every,
        ));
        if args.gateway.gateway_state_dir.is_some() {
            tokio::spawn(checkpoint_gateway::run_durable_loop(gateway.clone(), every));
        }
        tokio::spawn(checkpoint_gateway::run_checkpoint_loop(
            gateway,
            args.copy_url,
            proxy,
            make_proxy,
            Duration::from_secs(args.poll_interval),
            checkpoint_gateway::trainer_probe_limit(Duration::from_secs(args.request_timeout)),
        ));
    } else {
        const MAX_CONSECUTIVE_STORAGE_FULL: u32 = 5;
        let poll_interval = Duration::from_secs(args.poll_interval);
        let _poll_handle = tokio::spawn(async move {
            let mut storage_full_streak: u32 = 0;
            loop {
                match proxy.poll_and_download().await {
                    Ok(true) => {
                        storage_full_streak = 0;
                        info!("poll: new checkpoint downloaded successfully");
                    }
                    Ok(false) => {
                        storage_full_streak = 0;
                    }
                    Err(e) => {
                        let storage_full = e
                            .downcast_ref::<std::io::Error>()
                            .is_some_and(|io_err| io_err.kind() == std::io::ErrorKind::StorageFull);
                        if storage_full {
                            storage_full_streak += 1;
                            warn!(
                                "poll: storage full ({}/{}): {}",
                                storage_full_streak, MAX_CONSECUTIVE_STORAGE_FULL, e
                            );
                            if storage_full_streak >= MAX_CONSECUTIVE_STORAGE_FULL {
                                error!(
                                    "poll: {} consecutive storage-full polls; restarting to \
                                     release mmap'd tmpfs pages and start from a clean slate",
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
                tokio::time::sleep(poll_interval).await;
            }
        });
    }

    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("received SIGINT, shutting down");
        }
        result = server_handle => {
            if let Err(e) = result {
                error!("gRPC server error: {}", e);
            }
        }
        result = metrics_handle => {
            if let Err(e) = result {
                error!("metrics server error: {}", e);
            }
        }
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("checkpoint-server requires Linux (for CopyService and mmap)");
    std::process::exit(1);
}
