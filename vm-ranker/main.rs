use anyhow::{Context, Result};
use axum::Router;
use clap::Parser;
use log::info;
use std::sync::Arc;
use std::time::Duration;
use tonic::service::Routes;
use xai_http_server::{CancellationToken, GrpcConfig, HttpServer};

use xai_vm_ranker::{
    args::Args,
    carousel_model::CarouselModelStore,
    config_sync::{ConfigSync, ConfigSyncOptions},
    dpp::DppConfig,
    embedding_store,
    ranker_service::VMRankerServiceImpl,
    ranking_config::RankingConfig,
    scoring::DppContext,
    user_debias_store::UserDebiasStore,
};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    if args.enable_profiling {
        xai_profiling::spawn_server(3000, CancellationToken::new()).await;
    }

    let (dpp, preload_future) = if args.dpp_enabled {
        let (store, preload_future) = embedding_store::init_store(args.embedding_dim)
            .context("DPP requested but embedding store init failed")?;
        let config = DppConfig {
            top_k: args.dpp_top_k,
            theta: args.dpp_theta,
            max_selected_rank: args.dpp_max_selected_rank,
            debug_viewer_id: args.dpp_debug_viewer_id,
        };
        info!(
            "DPP enabled: top_k={}, theta={}, max_selected_rank={}, embedding_dim={}, debug_viewer_id={}",
            config.top_k,
            config.theta,
            config.max_selected_rank,
            args.embedding_dim,
            config.debug_viewer_id,
        );
        (Some(DppContext { store, config }), Some(preload_future))
    } else {
        info!("DPP rescoring disabled");
        (None, None)
    };

    let user_debias = args.user_debias_artifact_path.clone().map(|path| {
        let store = UserDebiasStore::load(path);
        store.spawn_reloader(Duration::from_secs(args.user_debias_reload_secs.max(1)));
        store
    });

    let process_overrides = xai_feature_switches::parse_fs_overrides(
        &std::env::var("XAI_FS_OVERRIDES").unwrap_or_default(),
    );
    if !process_overrides.is_empty() {
        info!(
            "process-wide feature-switch overrides (staging pins): {:?}",
            process_overrides.iter().map(|(k, _)| k).collect::<Vec<_>>()
        );
    }
    let ranking_config = if args.config_sync_enabled {
        let sync = ConfigSync::start(ConfigSyncOptions {
            remote: args.config_sync_remote.clone(),
            branch: args.config_sync_branch.clone(),
            root: args.config_sync_root.clone(),
            interval: Duration::from_secs(args.config_sync_interval_secs.max(5)),
            initial_sync_timeout: Duration::from_secs(args.config_sync_initial_timeout_secs),
        })
        .await
        .context("initial config sync failed")?;
        let config = Arc::new(
            RankingConfig::load(
                &ConfigSync::features_path(&args.config_sync_root),
                &ConfigSync::abdecider_path(&args.config_sync_root),
                process_overrides,
                args.fs_impressions_datacenter.as_deref(),
            )
            .await?,
        );
        info!(
            "ranking config loaded at revision {}; the feature-switch engine re-reads the synced files every 30s; impressions={}",
            sync.revision().unwrap_or_default(),
            args.fs_impressions_datacenter.as_deref().unwrap_or("off")
        );
        sync.spawn_poller();
        Some(config)
    } else {
        info!("config sync disabled; ranking parameters are not resolved");
        None
    };

    let carousel_models = args.carousel_model_dir.clone().map(|dir| {
        let store = CarouselModelStore::load(dir);
        store.spawn_reloader(Duration::from_secs(args.carousel_model_reload_secs.max(1)));
        store
    });
    let ranker_service = VMRankerServiceImpl::new(
        args.max_concurrent_requests,
        dpp,
        user_debias,
        ranking_config,
    )
    .with_carousel_models(carousel_models);
    info!(
        "Initialized VMRankerService with max_concurrent_requests={}",
        args.max_concurrent_requests
    );

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::NotServing)
        .await;
    let routes = Routes::new(ranker_service.server()).add_service(health_service);

    let grpc_config = GrpcConfig::new(args.grpc_port, routes);

    let mut http_server = HttpServer::builder(
        args.http_port,
        Router::new(),
        CancellationToken::new(),
        Duration::from_secs(10),
    )
    .with_grpc(grpc_config)
    .build()
    .await
    .context("Failed to create HTTP server")?;

    info!("HTTP server on port: {}", args.http_port);
    info!("gRPC server on port: {}", args.grpc_port);
    info!(
        "Metrics server on: http://0.0.0.0:{}/metrics",
        args.http_port
    );

    if let Some(preload) = preload_future {
        preload.await.context("O2 embedding preload failed")?;
    }

    http_server.set_readiness(true);
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    info!("HTTP/gRPC server is ready");

    http_server.wait_for_termination().await;
    info!("Server terminated");

    Ok(())
}
