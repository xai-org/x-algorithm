// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::{
    net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use axum::{extract::State, http, http::StatusCode, response::Html, routing::get};
use std::{future::Future, pin::Pin, sync::Mutex};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::{TcpListener, TcpSocket},
    signal::unix::{SignalKind, signal},
    sync::watch,
    task::{AbortHandle, JoinHandle},
};
pub use tokio_util::sync::CancellationToken;
use tonic::transport::server::TcpIncoming;
use tracing::{error, info, warn};

mod load_gauge;
mod readyz;
pub use load_gauge::{LoadGauge, LoadGuard};
use readyz::{ReadyzLayer, ReadyzProbe};

pub fn init_env_logger() {
    use env_logger::{Builder, Env};

    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        _ = Builder::from_env(Env::default().default_filter_or("warn"))
            .format_timestamp_millis()
            .format_line_number(true)
            .try_init()
            .ok();
    });
}

static XAI_HTTP_SERVER_FORCE_IPV6: std::sync::LazyLock<Option<bool>> =
    std::sync::LazyLock::new(|| {
        std::env::var("XAI_HTTP_SERVER_FORCE_IPV6")
            .ok()
            .and_then(|v| v.parse().ok())
    });

type StatusCallback = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NetConfig {
    #[default]
    Ipv4Only,
    Ipv6Only,
}

pub struct GrpcConfig<L = tower::layer::util::Identity> {
    pub port: u16,
    pub routes: tonic::service::Routes,
    pub tls_config: Option<tonic::transport::ServerTlsConfig>,
    pub concurrency_limit_per_connection: Option<usize>,
    pub http2_keepalive_interval: Option<Duration>,
    pub http2_keepalive_timeout: Option<Duration>,
    pub http2_max_local_error_reset_streams: Option<usize>,
    pub tcp_keepalive_time: Option<Duration>,
    pub max_connection_age: Option<Duration>,
    pub http2_max_header_list_size: Option<u32>,
    pub http2_max_concurrent_streams: Option<u32>,
    pub initial_stream_window_size: Option<u32>,
    pub initial_connection_window_size: Option<u32>,
    pub http2_adaptive_window: Option<bool>,
    pub accept_http1: bool,
    pub serve_readyz: bool,
    pub dscp: Option<u8>,
    pub layer: L,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            port: 0,
            routes: tonic::service::RoutesBuilder::default().routes(),
            tls_config: None,
            concurrency_limit_per_connection: None,
            http2_keepalive_interval: None,
            http2_keepalive_timeout: None,
            http2_max_local_error_reset_streams: None,
            tcp_keepalive_time: None,
            max_connection_age: None,
            http2_max_header_list_size: None,
            http2_max_concurrent_streams: None,
            initial_stream_window_size: None,
            initial_connection_window_size: None,
            http2_adaptive_window: None,
            accept_http1: false,
            serve_readyz: false,
            dscp: None,
            layer: tower::layer::util::Identity::new(),
        }
    }
}

impl GrpcConfig {
    pub fn new(port: u16, routes: tonic::service::Routes) -> Self {
        Self {
            port,
            routes,
            ..Default::default()
        }
    }
}
impl<L> GrpcConfig<L> {
    pub fn with_tls_config(mut self, tls_config: tonic::transport::ServerTlsConfig) -> Self {
        self.tls_config = Some(tls_config);
        self
    }

    pub fn with_layer<NewLayer>(
        self,
        new_layer: NewLayer,
    ) -> GrpcConfig<tower::layer::util::Stack<NewLayer, L>> {
        GrpcConfig {
            port: self.port,
            routes: self.routes,
            tls_config: self.tls_config,
            concurrency_limit_per_connection: self.concurrency_limit_per_connection,
            http2_keepalive_interval: self.http2_keepalive_interval,
            http2_keepalive_timeout: self.http2_keepalive_timeout,
            http2_max_local_error_reset_streams: self.http2_max_local_error_reset_streams,
            tcp_keepalive_time: self.tcp_keepalive_time,
            max_connection_age: self.max_connection_age,
            http2_max_header_list_size: self.http2_max_header_list_size,
            http2_max_concurrent_streams: self.http2_max_concurrent_streams,
            initial_stream_window_size: self.initial_stream_window_size,
            initial_connection_window_size: self.initial_connection_window_size,
            http2_adaptive_window: self.http2_adaptive_window,
            accept_http1: self.accept_http1,
            serve_readyz: self.serve_readyz,
            dscp: self.dscp,
            layer: tower::layer::util::Stack::new(new_layer, self.layer),
        }
    }

    pub fn accept_http1(mut self, accept: bool) -> Self {
        self.accept_http1 = accept;
        self
    }

    pub fn with_readyz_endpoint(mut self) -> Self {
        self.serve_readyz = true;
        self
    }

    pub fn with_max_connection_age(mut self, duration: Duration) -> Self {
        self.max_connection_age = Some(duration);
        self
    }

    pub fn with_http2_max_header_list_size(mut self, size: u32) -> Self {
        self.http2_max_header_list_size = Some(size);
        self
    }

    pub fn with_dscp(mut self, dscp: u8) -> Self {
        self.dscp = Some(dscp);
        self
    }
}

pub struct HttpMtlsConfig {
    pub port: u16,
    pub router: axum::Router,
    pub tls_config: Option<std::sync::Arc<rustls::ServerConfig>>,
}

impl HttpMtlsConfig {
    pub fn new(port: u16, router: axum::Router) -> Self {
        Self {
            port,
            router,
            tls_config: None,
        }
    }

    pub fn with_tls(mut self, config: std::sync::Arc<rustls::ServerConfig>) -> Self {
        self.tls_config = Some(config);
        self
    }
}

#[derive(Debug)]
pub struct HttpServer {
    is_accepting: Arc<AtomicBool>,
    rx_terminated: watch::Receiver<bool>,
    abort_http: AbortHandle,
    abort_grpc: AbortHandle,
    abort_http_mtls: AbortHandle,
    cancel: CancellationToken,
    server_state: ServerState,
    pub http_port: u16,
    pub grpc_port: Option<u16>,
    pub http_mtls_port: Option<u16>,
}

pub struct HttpServerBuilder<L = tower::layer::util::Identity> {
    http_port: u16,
    http_router: axum::Router,
    grpc: Option<GrpcConfig<L>>,
    http_mtls: Option<HttpMtlsConfig>,
    cancel: CancellationToken,
    termination_delay: Duration,
    net: NetConfig,
    so_reuseaddr: bool,
}

impl<L> HttpServerBuilder<L> {
    pub fn with_grpc<NewL>(self, grpc: GrpcConfig<NewL>) -> HttpServerBuilder<NewL> {
        HttpServerBuilder {
            http_port: self.http_port,
            http_router: self.http_router,
            grpc: Some(grpc),
            http_mtls: self.http_mtls,
            cancel: self.cancel,
            termination_delay: self.termination_delay,
            net: self.net,
            so_reuseaddr: self.so_reuseaddr,
        }
    }

    pub fn with_net_config(mut self, net: NetConfig) -> Self {
        self.net = net;
        self
    }

    pub fn with_http_mtls(mut self, config: HttpMtlsConfig) -> Self {
        self.http_mtls = Some(config);
        self
    }

    pub fn with_so_reuseaddr(mut self, enabled: bool) -> Self {
        self.so_reuseaddr = enabled;
        self
    }

    pub async fn build(self) -> std::io::Result<HttpServer>
    where
        L: tower::Layer<tonic::service::Routes> + Clone + Send + 'static,
        L::Service: tower::Service<
                http::Request<tonic::body::Body>,
                Response = http::Response<tonic::body::Body>,
            > + Clone
            + Send
            + 'static,
        <L::Service as tower::Service<http::Request<tonic::body::Body>>>::Future: Send + 'static,
        <L::Service as tower::Service<http::Request<tonic::body::Body>>>::Error:
            Into<Box<dyn std::error::Error + Send + Sync>> + Send,
    {
        HttpServer::new_internal(
            self.http_port,
            self.http_router,
            self.grpc,
            self.http_mtls,
            self.cancel,
            self.termination_delay,
            self.net,
            self.so_reuseaddr,
        )
        .await
    }
}

impl HttpServer {
    pub fn builder(
        http_port: u16,
        http_router: axum::Router,
        cancel: CancellationToken,
        termination_delay: Duration,
    ) -> HttpServerBuilder {
        let net = match *XAI_HTTP_SERVER_FORCE_IPV6 {
            Some(true) => NetConfig::Ipv6Only,
            _ => NetConfig::default(),
        };
        HttpServerBuilder {
            http_port,
            http_router,
            grpc: None,
            http_mtls: None,
            cancel,
            termination_delay,
            net,
            so_reuseaddr: true,
        }
    }

    async fn new_internal<L>(
        http_port: u16,
        http_router: axum::Router,
        grpc: Option<GrpcConfig<L>>,
        http_mtls: Option<HttpMtlsConfig>,
        cancel: CancellationToken,
        termination_delay: Duration,
        net: NetConfig,
        so_reuseaddr: bool,
    ) -> std::io::Result<Self>
    where
        L: tower::Layer<tonic::service::Routes> + Clone + Send + 'static,
        L::Service: tower::Service<
                http::Request<tonic::body::Body>,
                Response = http::Response<tonic::body::Body>,
            > + Clone
            + Send
            + 'static,
        <L::Service as tower::Service<http::Request<tonic::body::Body>>>::Future: Send + 'static,
        <L::Service as tower::Service<http::Request<tonic::body::Body>>>::Error:
            Into<Box<dyn std::error::Error + Send + Sync>> + Send,
    {
        let server_state = ServerState::new();
        spawn_cancellation_on_sigterm(
            cancel.clone(),
            termination_delay,
            server_state.is_terminating.clone(),
        );
        let is_accepting = server_state.is_accepting.clone();
        let (http_handle, http_port) = {
            let http_listener = bind_listener(http_port, net, so_reuseaddr, None)
                .map_err(|e| map_bind_error(e, "http", http_port))?;
            let http_port = http_listener.local_addr()?.port();
            let http_handle = spawn_http_server(
                http_listener,
                cancel.clone(),
                http_router,
                server_state.clone(),
            );
            (http_handle, http_port)
        };
        let (tx_terminated, rx_terminated) = watch::channel(false);
        let (grpc_handle, grpc_port) = if let Some(grpc) = grpc {
            let grpc_listener = bind_listener(grpc.port, net, so_reuseaddr, grpc.dscp)
                .map_err(|e| map_bind_error(e, "grpc", grpc.port))?;
            let grpc_port = grpc_listener.local_addr()?.port();
            let grpc_handle =
                spawn_grpc_server(grpc_listener, grpc, cancel.clone(), server_state.clone());
            (grpc_handle, Some(grpc_port))
        } else {
            (tokio::spawn(async {}), None)
        };
        let (http_mtls_handle, http_mtls_port) = if let Some(http_mtls) = http_mtls {
            let (handle, port) =
                spawn_http_mtls_server(http_mtls, cancel.clone(), net, so_reuseaddr)?;
            (handle, Some(port))
        } else {
            (tokio::spawn(async {}), None)
        };
        let abort_http = http_handle.abort_handle();
        let abort_grpc = grpc_handle.abort_handle();
        let abort_http_mtls = http_mtls_handle.abort_handle();
        tokio::spawn(async move {
            grpc_handle.await.ok();
            http_handle.await.ok();
            http_mtls_handle.await.ok();
            tx_terminated.send(true).ok();
        });
        Ok(Self {
            abort_grpc,
            abort_http,
            abort_http_mtls,
            is_accepting,
            rx_terminated,
            cancel,
            server_state,
            http_port,
            grpc_port,
            http_mtls_port,
        })
    }

    pub fn set_readiness(&self, is_ready: bool) {
        self.is_accepting.store(is_ready, Ordering::Relaxed);
    }

    pub fn terminate_gracefully(&self) {
        self.cancel.cancel();
    }

    pub fn terminate_forcefully(&self) {
        self.abort_grpc.abort();
        self.abort_http.abort();
        self.abort_http_mtls.abort();
    }

    pub fn is_terminated(&self) -> bool {
        *self.rx_terminated.borrow()
    }

    pub async fn wait_for_termination(&mut self) {
        self.rx_terminated.wait_for(|x| *x).await.unwrap();
    }

    pub fn add_statusz_callback<F, Fut>(&self, f: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        self.server_state.add_statusz_callback(f);
    }
}

fn map_bind_error(e: std::io::Error, name: &str, port: u16) -> std::io::Error {
    std::io::Error::new(
        e.kind(),
        format!("Failed to start {name} server on port {port}: {e}"),
    )
}

fn bind_addr(port: u16, net: NetConfig) -> SocketAddr {
    match net {
        NetConfig::Ipv4Only => SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port)),
        NetConfig::Ipv6Only => SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, port, 0, 0)),
    }
}

fn bind_listener(
    port: u16,
    net: NetConfig,
    so_reuseaddr: bool,
    dscp: Option<u8>,
) -> std::io::Result<TcpListener> {
    let addr = bind_addr(port, net);
    let socket = match addr {
        SocketAddr::V4(_) => TcpSocket::new_v4()?,
        SocketAddr::V6(_) => TcpSocket::new_v6()?,
    };
    if so_reuseaddr {
        socket.set_reuseaddr(true)?;
    }
    if let Some(dscp) = dscp {
        if dscp > 63 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("dscp must be 0-63, got {dscp}"),
            ));
        }
        let tos = u32::from(dscp) << 2;
        match addr {
            SocketAddr::V4(_) => socket.set_tos_v4(tos)?,
            SocketAddr::V6(_) => {
                socket.set_tclass_v6(tos)?;
                #[cfg(target_os = "linux")]
                socket.set_tos_v4(tos)?;
            }
        }
    }
    socket.bind(addr)?;
    socket.listen(1024)
}

fn spawn_cancellation_on_sigterm(
    cancel: CancellationToken,
    termination_delay: Duration,
    is_terminating: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        if signal(SignalKind::terminate())
            .unwrap()
            .recv()
            .await
            .is_some()
        {
            info!("SIGTERM signal received, triggering cancellation in {termination_delay:?}");
            is_terminating.store(true, Ordering::Relaxed);
            tokio::time::sleep(termination_delay).await;
            cancel.cancel();
        }
    });
}

fn spawn_grpc_server<L>(
    listener: TcpListener,
    grpc: GrpcConfig<L>,
    cancel: CancellationToken,
    server_state: ServerState,
) -> JoinHandle<()>
where
    L: tower::Layer<tonic::service::Routes> + Clone + Send + 'static,
    L::Service: tower::Service<
            http::Request<tonic::body::Body>,
            Response = http::Response<tonic::body::Body>,
        > + Clone
        + Send
        + 'static,
    <L::Service as tower::Service<http::Request<tonic::body::Body>>>::Future: Send + 'static,
    <L::Service as tower::Service<http::Request<tonic::body::Body>>>::Error:
        Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    tokio::spawn(async move {
        let GrpcConfig {
            routes,
            tls_config,
            concurrency_limit_per_connection,
            http2_keepalive_interval,
            http2_keepalive_timeout,
            tcp_keepalive_time,
            max_connection_age,
            http2_max_header_list_size,
            http2_max_concurrent_streams,
            initial_stream_window_size,
            initial_connection_window_size,
            http2_adaptive_window,
            accept_http1,
            serve_readyz,
            layer,
            ..
        } = grpc;

        let readyz_layer = ReadyzLayer::new(serve_readyz.then(|| ReadyzProbe {
            is_accepting: server_state.is_accepting.clone(),
            is_terminating: server_state.is_terminating.clone(),
        }));

        let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
        let mut server = tonic::transport::server::Server::builder().accept_http1(accept_http1);

        if let Some(t) = http2_keepalive_interval {
            server = server.http2_keepalive_interval(Some(t));
        }
        if let Some(t) = http2_keepalive_timeout {
            server = server.http2_keepalive_timeout(Some(t));
        }
        if let Some(n) = concurrency_limit_per_connection {
            server = server.concurrency_limit_per_connection(n);
        }
        if let Some(t) = tcp_keepalive_time {
            server = server.tcp_keepalive(Some(t));
        }
        if let Some(t) = max_connection_age {
            server = server.max_connection_age(t);
        }
        if let Some(size) = http2_max_header_list_size {
            server = server.http2_max_header_list_size(size);
        }
        if let Some(max) = http2_max_concurrent_streams {
            server = server.max_concurrent_streams(Some(max));
        }
        if let Some(size) = initial_stream_window_size {
            server = server.initial_stream_window_size(Some(size));
        }
        if let Some(size) = initial_connection_window_size {
            server = server.initial_connection_window_size(Some(size));
        }
        if let Some(enabled) = http2_adaptive_window {
            server = server.http2_adaptive_window(Some(enabled));
        }

        if let Some(tls_config) = tls_config {
            server = match server.tls_config(tls_config) {
                Ok(s) => s,
                Err(e) => {
                    error!("grpc tls config failed: {e}");
                    std::process::exit(1);
                }
            };
        }
        let status = server
            .layer(readyz_layer)
            .layer(fastrace_tonic::FastraceServerLayer)
            .layer(layer)
            .add_routes(routes)
            .serve_with_incoming_shutdown(incoming, cancel.cancelled())
            .await;
        info!("grpc server terminated with status: {status:?}");
        fastrace::flush();
    })
}

const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

struct TlsListener {
    inner: TcpListener,
    acceptor: tokio_rustls::TlsAcceptor,
}

struct TlsConn {
    stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
}

impl AsyncRead for TlsConn {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsConn {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }
    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsConn;
    type Addr = SocketAddr;

    fn accept(&mut self) -> impl Future<Output = (Self::Io, Self::Addr)> + Send {
        let inner = &mut self.inner;
        let acceptor = self.acceptor.clone();
        async move {
            loop {
                let (tcp_stream, peer_addr) = inner.accept().await;

                match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, acceptor.accept(tcp_stream)).await
                {
                    Ok(Ok(tls_stream)) => {
                        return (TlsConn { stream: tls_stream }, peer_addr);
                    }
                    Ok(Err(e)) => {
                        warn!("TLS handshake failed from {peer_addr}: {e}");
                    }
                    Err(_) => {
                        warn!(
                            "TLS handshake timed out from {peer_addr} \
                             (client may not have sent a certificate)"
                        );
                    }
                }
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

fn spawn_http_mtls_server(
    http_mtls: HttpMtlsConfig,
    cancel: CancellationToken,
    net: NetConfig,
    so_reuseaddr: bool,
) -> std::io::Result<(JoinHandle<()>, u16)> {
    info!(
        "Starting application HTTP mTLS server on port {}. has_tls: {}",
        http_mtls.port,
        http_mtls.tls_config.is_some()
    );
    let listener = bind_listener(http_mtls.port, net, so_reuseaddr, None)
        .map_err(|e| map_bind_error(e, "http-mtls", http_mtls.port))?;
    let port = listener.local_addr()?.port();

    if let Some(tls_config) = http_mtls.tls_config {
        let tls_listener = TlsListener {
            inner: listener,
            acceptor: tokio_rustls::TlsAcceptor::from(tls_config),
        };

        let jh = tokio::spawn(async move {
            let status = axum::serve(tls_listener, http_mtls.router.into_make_service())
                .with_graceful_shutdown(cancel.cancelled_owned())
                .await;
            info!("app http (tls) server terminated: {status:?}");
        });

        Ok((jh, port))
    } else {
        let jh = tokio::spawn(async move {
            let status = axum::serve(listener, http_mtls.router.into_make_service())
                .with_graceful_shutdown(cancel.cancelled_owned())
                .await;
            info!("app http server terminated: {status:?}");
        });

        Ok((jh, port))
    }
}

fn spawn_http_server(
    listener: TcpListener,
    cancel: CancellationToken,
    router: axum::Router,
    server_state: ServerState,
) -> JoinHandle<()> {
    let rest = axum::Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .route("/readyz", get(readyz))
        .route("/defaultstatusz", get(statusz_handler))
        .with_state(server_state);
    let router = router.merge(rest);
    tokio::spawn(async move {
        let status = axum::serve(listener, router.into_make_service())
            .with_graceful_shutdown(cancel.cancelled_owned())
            .await;
        info!("http server terminated with status: {status:?}");
    })
}

async fn healthz() -> (StatusCode, &'static str) {
    (StatusCode::OK, "Healthy")
}

async fn metrics() -> (StatusCode, String) {
    let report = prometheus::default_registry().gather();
    let result = prometheus::TextEncoder::new().encode_to_string(&report);
    match result {
        Ok(x) => (StatusCode::OK, x),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to format Prometheus metrics: {e}"),
        ),
    }
}

async fn readyz(State(server_state): State<ServerState>) -> (StatusCode, &'static str) {
    if server_state.is_accepting.load(Ordering::Acquire)
        && !server_state.is_terminating.load(Ordering::Acquire)
    {
        (StatusCode::OK, "Ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Not Ready")
    }
}

pub fn default_statusz_content() -> String {
    let build_info = "xai-recsys-engine (oss)";
    let service_name = std::env::current_exe()
        .ok()
        .and_then(|path| path.file_name()?.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        r#"<!DOCTYPE html>
    <html>
    <head>
        <title>{service_name} Status</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>{service_name} Status</h1>
        <div style="display: flex; align-items: center;">
            <p><strong>Build:</strong> {build_info}</p>
        </div>
    "#,
    )
}

#[derive(Clone)]
struct ServerState {
    is_terminating: Arc<AtomicBool>,
    is_accepting: Arc<AtomicBool>,
    callbacks: Arc<Mutex<Vec<StatusCallback>>>,
}

impl std::fmt::Debug for ServerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerState")
            .field("is_terminating", &self.is_terminating)
            .field("is_accepting", &self.is_accepting)
            .field(
                "callbacks_count",
                &self.callbacks.lock().map(|c| c.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl ServerState {
    fn new() -> Self {
        Self {
            is_terminating: Arc::new(AtomicBool::default()),
            is_accepting: Arc::new(AtomicBool::default()),
            callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn add_statusz_callback<F, Fut>(&self, f: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        let wrapped: StatusCallback = Arc::new(move || Box::pin(f()));
        if let Ok(mut guard) = self.callbacks.lock() {
            guard.push(wrapped);
        }
    }

    async fn get_extra_content(&self) -> String {
        let cbs = {
            if let Ok(guard) = self.callbacks.lock() {
                guard.clone()
            } else {
                Vec::new()
            }
        };
        let mut extra = String::new();
        for cb in cbs {
            extra.push_str(&cb().await);
        }
        extra
    }
}

async fn statusz_handler(State(state): State<ServerState>) -> Html<String> {
    let content = default_statusz_content();
    let extra = state.get_extra_content().await;
    let html = if extra.is_empty() {
        format!("{} </body></html>", content)
    } else {
        format!("{} {} </body></html>", content, extra)
    };
    Html(html)
}

#[cfg(test)]
mod test {
    use super::*;
    use tokio::net::TcpStream;

    #[tokio::test]
    async fn dscp_marks_accepted_connections() {
        let listener = bind_listener(0, NetConfig::Ipv4Only, true, Some(1)).unwrap();
        let addr = listener.local_addr().unwrap();
        let (_client, accepted) = tokio::join!(TcpStream::connect(addr), listener.accept());
        let accepted = TcpSocket::from_std_stream(accepted.unwrap().0.into_std().unwrap());
        assert_eq!(accepted.tos_v4().unwrap(), 1 << 2);
    }

    #[tokio::test]
    async fn dscp_out_of_range_is_rejected() {
        let err = bind_listener(0, NetConfig::Ipv4Only, true, Some(64)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
