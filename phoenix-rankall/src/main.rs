use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use log::info;
use tokio_util::sync::CancellationToken;

use xai_recsys_rankall::config::{Cli, PipelineKind};
use xai_recsys_rankall::consumer::KafkaEventConsumer;
use xai_recsys_rankall::metrics::PipelineMetrics;
use xai_recsys_rankall::pipeline::Pipeline;
use xai_recsys_rankall::processor::analysis_processor::AnalysisProcessor;
use xai_recsys_rankall::processor::blacklist::Blacklist;
use xai_recsys_rankall::processor::main_processor::MainProcessor;
use xai_recsys_rankall::processor::metadata_processor::MetadataProcessor;
use xai_recsys_rankall::processor::sid_popular_video_processor::SidPopularVideoProcessor;
use xai_recsys_rankall::processor::sid_processor::SidProcessor;
use xai_recsys_rankall::processor::sid_tail_processor::SidTailProcessor;
use xai_recsys_rankall::processor::topic_processor::TopicProcessor;
use xai_recsys_rankall::sid::backfill::{spawn_backfill_loop, BackfillConfig};
use xai_recsys_rankall::sid::client::SidClient;
use xai_recsys_rankall::sid::metrics::SidMetrics;
use xai_recsys_rankall::store::analysis::AnalysisDumpStore;
use xai_recsys_rankall::store::base::{
    metadata_window_router, prefix_window_router, sid_window_router, topic_window_router,
    BaseSnapshotStore,
};
use xai_recsys_rankall::store::StoreConfig;

#[tokio::main]
async fn main() -> Result<()> {
    xai_init_utils::init().log();

    let cli = Cli::parse();
    info!("starting xai-recsys-rankall: pipeline={}", cli.pipeline);

    if !cli.pipeline.is_implemented() {
        bail!(
            "pipeline variant '{}' is not yet implemented.",
            cli.pipeline
        );
    }

    let registry = prometheus::default_registry().clone();
    let metrics = PipelineMetrics::new_with_pipeline(registry.clone(), &cli.pipeline.to_string())
        .context("failed to create metrics")?;
    let metrics = Arc::new(metrics);

    let cancel = CancellationToken::new();

    let http_server = xai_http_server::HttpServer::builder(
        cli.metrics_port,
        axum::Router::new(),
        cancel.clone(),
        Duration::ZERO,
    )
    .build()
    .await
    .context("failed to start HTTP server")?;
    http_server.set_readiness(true);
    info!(
        "HTTP health/metrics server started on port {}",
        cli.metrics_port
    );

    info!(
        "building pipeline: variant={}, topic={}, group={}, output_dir={}",
        cli.pipeline,
        cli.effective_topic(),
        cli.group,
        cli.output_dir.display(),
    );

    let mut sid_backfill_handle: Option<tokio::task::JoinHandle<()>> = None;

    let pipeline = match cli.pipeline {
        PipelineKind::Main => {
            let consumer = KafkaEventConsumer::from_cli(&cli).await?;
            let processor = MainProcessor::new();
            let store = BaseSnapshotStore::new(
                StoreConfig {
                    output_dir: cli.output_dir.clone(),
                    windows: cli.pipeline.window_configs(),
                    compaction_interval_secs: cli.dump_interval_secs,
                    versions_to_keep: cli.versions_to_keep,
                    pipeline: cli.pipeline.to_string(),
                    sid_num_levels: cli.sid_num_levels,
                },
                prefix_window_router(),
                Arc::clone(&metrics),
            );
            Pipeline::new(
                Box::new(consumer),
                Box::new(processor),
                Box::new(store),
                Arc::try_unwrap(metrics).unwrap_or_else(|m| (*m).clone()),
            )
        }
        PipelineKind::Topic => {
            let consumer = KafkaEventConsumer::from_cli(&cli).await?;
            let blacklist = Blacklist::load(&cli.output_dir);
            let processor = TopicProcessor::new(blacklist);
            let store = BaseSnapshotStore::new(
                StoreConfig {
                    output_dir: cli.output_dir.clone(),
                    windows: cli.pipeline.window_configs(),
                    compaction_interval_secs: cli.dump_interval_secs,
                    versions_to_keep: cli.versions_to_keep,
                    pipeline: cli.pipeline.to_string(),
                    sid_num_levels: cli.sid_num_levels,
                },
                topic_window_router(),
                Arc::clone(&metrics),
            );
            Pipeline::new(
                Box::new(consumer),
                Box::new(processor),
                Box::new(store),
                Arc::try_unwrap(metrics).unwrap_or_else(|m| (*m).clone()),
            )
        }
        PipelineKind::Metadata => {
            let consumer = KafkaEventConsumer::from_cli(&cli).await?;
            let processor = MetadataProcessor::new("metadata");
            let store = BaseSnapshotStore::new(
                StoreConfig {
                    output_dir: cli.output_dir.clone(),
                    windows: cli.pipeline.window_configs(),
                    compaction_interval_secs: cli.dump_interval_secs,
                    versions_to_keep: cli.versions_to_keep,
                    pipeline: cli.pipeline.to_string(),
                    sid_num_levels: cli.sid_num_levels,
                },
                metadata_window_router(),
                Arc::clone(&metrics),
            );
            Pipeline::new(
                Box::new(consumer),
                Box::new(processor),
                Box::new(store),
                Arc::try_unwrap(metrics).unwrap_or_else(|m| (*m).clone()),
            )
        }
        PipelineKind::Sid | PipelineKind::SidTail | PipelineKind::SidPopularVideo => {
            let endpoint = cli.sid_endpoint.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--sid-endpoint is required when --pipeline={}",
                    cli.pipeline
                )
            })?;
            if cli.sid_num_levels == 0 {
                bail!("--sid-num-levels must be > 0 for the SID pipeline");
            }
            let mut windows = cli.pipeline.window_configs();
            if cli.pipeline == PipelineKind::SidPopularVideo {
                windows = windows
                    .into_iter()
                    .map(|w| {
                        w.excluding_posts_in_file(&cli.popular_video_exclude_posts_in)
                            .requiring_posts_in_file(&cli.popular_video_require_posts_in)
                    })
                    .collect();
            }

            let sid_metrics = SidMetrics::new(&registry).context("failed to create SID metrics")?;
            let sid_client = Arc::new(
                SidClient::new(
                    endpoint,
                    cli.sid_min_post_age_seconds,
                    Duration::from_secs_f64(cli.sid_request_timeout_seconds),
                    sid_metrics.clone(),
                )
                .context("failed to construct SidClient")?,
            );
            let window_names: Vec<String> = windows.iter().map(|w| w.window_name()).collect();

            let consumer = KafkaEventConsumer::from_cli(&cli).await?;
            let processor: Box<dyn xai_recsys_rankall::processor::RecordProcessor> =
                match cli.pipeline {
                    PipelineKind::Sid => {
                        let index_filters: std::collections::HashSet<String> =
                            windows.iter().map(|w| w.name.clone()).collect();
                        info!(
                            "SID hydration enabled: endpoint={endpoint}, min_post_age={}s, \
                             timeout={}s, backfill_max_age={}h, windows={window_names:?}, \
                             index_filter={index_filters:?}",
                            cli.sid_min_post_age_seconds,
                            cli.sid_request_timeout_seconds,
                            cli.sid_backfill_max_age_hours,
                        );
                        Box::new(SidProcessor::new(
                            index_filters,
                            Arc::clone(&sid_client),
                            tokio::runtime::Handle::current(),
                        ))
                    }
                    PipelineKind::SidTail => {
                        info!(
                            "SID-tail hydration enabled: endpoint={endpoint}, \
                             max_author_followers={}, min_fav_count={}, min_post_age={}s, \
                             timeout={}s, backfill_max_age={}h, windows={window_names:?}",
                            cli.tail_max_author_followers,
                            cli.tail_min_fav_count,
                            cli.sid_min_post_age_seconds,
                            cli.sid_request_timeout_seconds,
                            cli.sid_backfill_max_age_hours,
                        );
                        Box::new(SidTailProcessor::new(
                            cli.tail_max_author_followers,
                            cli.tail_min_fav_count,
                            Arc::clone(&sid_client),
                            tokio::runtime::Handle::current(),
                        ))
                    }
                    PipelineKind::SidPopularVideo => {
                        info!(
                            "SID popular-video hydration enabled: endpoint={endpoint}, \
                             min_view_count={}, min_fav_count={}, exclude_posts_in={}, \
                             require_posts_in={}, min_post_age={}s, timeout={}s, \
                             backfill_max_age={}h, windows={window_names:?}",
                            cli.popular_video_min_view_count,
                            cli.popular_video_min_fav_count,
                            cli.popular_video_exclude_posts_in.display(),
                            cli.popular_video_require_posts_in.display(),
                            cli.sid_min_post_age_seconds,
                            cli.sid_request_timeout_seconds,
                            cli.sid_backfill_max_age_hours,
                        );
                        Box::new(SidPopularVideoProcessor::new(
                            cli.popular_video_min_view_count,
                            cli.popular_video_min_fav_count,
                            Arc::clone(&sid_client),
                            tokio::runtime::Handle::current(),
                        ))
                    }
                    _ => unreachable!(),
                };

            let store = Arc::new(BaseSnapshotStore::new(
                StoreConfig {
                    output_dir: cli.output_dir.clone(),
                    windows,
                    compaction_interval_secs: cli.dump_interval_secs,
                    versions_to_keep: cli.versions_to_keep,
                    pipeline: cli.pipeline.to_string(),
                    sid_num_levels: cli.sid_num_levels,
                },
                sid_window_router(),
                Arc::clone(&metrics),
            ));

            sid_backfill_handle = Some(spawn_backfill_loop(
                Arc::clone(&store),
                Arc::clone(&sid_client),
                sid_metrics,
                BackfillConfig {
                    interval: Duration::from_secs(cli.sid_backfill_interval_secs),
                    batch_size: cli.sid_backfill_batch_size,
                    max_per_cycle: cli.sid_backfill_qps_cap,
                    max_age_hours: cli.sid_backfill_max_age_hours,
                },
                cancel.clone(),
            ));

            Pipeline::new(
                Box::new(consumer),
                processor,
                Box::new(store),
                Arc::try_unwrap(metrics).unwrap_or_else(|m| (*m).clone()),
            )
        }
        PipelineKind::Analysis => {
            let consumer = KafkaEventConsumer::from_cli(&cli).await?;
            let processor = AnalysisProcessor::new(cli.score_prefix.clone(), cli.compact_protocol);
            let store = AnalysisDumpStore::new(cli.output_dir.clone(), None);
            Pipeline::new(
                Box::new(consumer),
                Box::new(processor),
                Box::new(store),
                Arc::try_unwrap(metrics).unwrap_or_else(|m| (*m).clone()),
            )
        }
        _ => bail!("pipeline variant '{}' is not implemented", cli.pipeline),
    };

    let pipeline: Pipeline =
        pipeline.with_commit_strategy(cli.commit_after_dump, cli.commit_interval_secs);

    info!(
        "pipeline built (commit_after_dump={}, commit_interval={}s), starting main loop",
        cli.commit_after_dump, cli.commit_interval_secs,
    );
    let result = pipeline.run(cancel).await;

    if let Some(handle) = sid_backfill_handle {
        let _ = handle.await;
    }
    result
}
