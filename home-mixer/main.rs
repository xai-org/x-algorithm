use std::time::Duration;

use clap::Parser;
use xai_candidate_pipeline::component_library::utils::quality_factor;
use xai_dark_traffic::RejectDarkTrafficLayer;
use xai_home_mixer::dark_traffic_setup;
use xai_home_mixer::params;
use xai_home_mixer::{HomeMixerConfig, HomeMixerServer, PhoenixXdsConfig, VmRankerXdsConfig};
use xai_home_mixer_proto as pb;
use xai_x_rpc::grpc_client::TlsMode;
use xai_x_rpc::wily_lookup_service::ShardCoordinate;
use xai_x_service_builder::XServiceBuilder;

#[derive(Parser, Debug)]
#[command(about = "HomeMixer gRPC Server")]
struct Args {
    #[arg(long, default_value_t = 8080u16)]
    grpc_port: u16,
    #[arg(long, default_value_t = 9090u16)]
    metrics_port: u16,
    #[arg(long, default_value_t = -1)]
    shard_coordinate: i16,
    #[arg(long, default_value_t = 500)]
    shard_total_size: u16,
    #[arg(long, default_value = "atla")]
    datacenter: String,
    #[arg(long, default_value = "")]
    otel_endpoint: String,
    #[arg(long)]
    popular_authors_job: bool,
    #[arg(long)]
    popular_authors_job_once: bool,
    #[arg(long, default_value_t = 3600)]
    popular_authors_job_interval_secs: u64,
    #[arg(long, default_value_t = xai_home_mixer::util::popular_authors::TOP_POSTING_AUTHORS_FRACTION)]
    popular_authors_fraction: f64,
    #[arg(
        long,
        default_value = "twttr-bq-timelines-prod.pulse.popular_posting_authors"
    )]
    popular_authors_bigquery_table: String,
    #[arg(long, default_value = "/etc/pulse-bq/key.json")]
    popular_authors_bigquery_key_path: String,
    #[arg(long)]
    popular_authors_egress_proxy: Option<String>,
    #[arg(long, default_value_t = 30)]
    popular_authors_max_snapshot_age_hours: u64,
    #[arg(long)]
    popular_posts_job: bool,
    #[arg(long)]
    popular_posts_job_once: bool,
    #[arg(long, default_value_t = 300)]
    popular_posts_job_interval_secs: u64,
    #[arg(long, default_value_t = 5)]
    popular_posts_per_author: usize,
    #[arg(long, default_value_t = 500)]
    popular_posts_budget: usize,
    #[arg(long, default_value_t = 8.0)]
    popular_posts_half_life_hours: f64,
    #[arg(long)]
    select_popular_posts: bool,
    #[arg(long, value_parser = ["authors", "posts"])]
    dump_popular_store: Option<String>,

    #[arg(long, default_value = "penalized_peak_ewma")]
    phoenix_xds_lb_policy: String,
    #[arg(long, default_value = "penalized_peak_ewma")]
    phoenix_xds_retrieval_lb_policy: String,
    #[arg(long, default_value_t = 500)]
    phoenix_xds_lb_default_rtt_ms: u64,
    #[arg(long, default_value_t = 4194304)]
    phoenix_xds_h2_stream_window_bytes: u32,
    #[arg(long, default_value_t = 16777216)]
    phoenix_xds_h2_connection_window_bytes: u32,
    #[arg(long, default_value_t = 4194304)]
    phoenix_xds_socket_buffer_bytes: u32,
    #[arg(long, default_value_t = 12)]
    phoenix_xds_aperture_size: u32,
    #[arg(long, default_value = "readiness")]
    phoenix_xds_health_probe: String,
    #[arg(long, default_value_t = 1500)]
    phoenix_xds_grpc_health_timeout_ms: u64,
    #[arg(long, default_value_t = 3000)]
    phoenix_xds_grpc_health_interval_ms: u64,
    #[arg(long, default_value_t = 9091)]
    phoenix_xds_readiness_port: u16,
    #[arg(long, default_value = "/readyz")]
    phoenix_xds_readiness_path: String,
    #[arg(long, default_value = "atla")]
    phoenix_xds_discovery_authority: String,

    #[arg(long, default_value = "tonic")]
    vm_ranker_xds_lb_policy: String,
    #[arg(long, default_value_t = 500)]
    vm_ranker_xds_lb_default_rtt_ms: u64,
    #[arg(long, default_value_t = 4194304)]
    vm_ranker_xds_h2_stream_window_bytes: u32,
    #[arg(long, default_value_t = 16777216)]
    vm_ranker_xds_h2_connection_window_bytes: u32,
    #[arg(long, default_value_t = 4194304)]
    vm_ranker_xds_socket_buffer_bytes: u32,
    #[arg(long, default_value_t = 12)]
    vm_ranker_xds_aperture_size: u32,
    #[arg(long, default_value = "readiness")]
    vm_ranker_xds_health_probe: String,
    #[arg(long, default_value_t = 1500)]
    vm_ranker_xds_grpc_health_timeout_ms: u64,
    #[arg(long, default_value_t = 3000)]
    vm_ranker_xds_grpc_health_interval_ms: u64,
    #[arg(long, default_value_t = 8080)]
    vm_ranker_xds_readiness_port: u16,
    #[arg(long, default_value = "/healthz")]
    vm_ranker_xds_readiness_path: String,
    #[arg(long, default_value = "atla")]
    vm_ranker_xds_discovery_authority: String,
}

fn parse_shard(args: &Args) -> Option<ShardCoordinate> {
    if args.shard_coordinate >= 0 {
        Some(ShardCoordinate {
            ordinal: args.shard_coordinate as u16,
            total_size: args.shard_total_size,
        })
    } else {
        None
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(store) = &args.dump_popular_store {
        return dump_popular_store(&args.datacenter, store == "posts").await;
    }
    if args.popular_authors_job || args.popular_authors_job_once {
        let _stats =
            xai_stats_receiver::init_stats_receiver_guarded("home-mixer-popular-authors-job");
        let _tracing = xai_pipeline_tracing::init_tracing(
            "xai-home-mixer-popular-authors-job",
            &args.otel_endpoint,
        );
        xai_init_utils::init().rustls();
        jsonwebtoken::crypto::aws_lc::DEFAULT_PROVIDER
            .install_default()
            .ok();
        return xai_home_mixer::popular_authors_job::run(
            xai_home_mixer::popular_authors_job::JobConfig {
                datacenter: args.datacenter.clone(),
                interval: Duration::from_secs(args.popular_authors_job_interval_secs),
                fraction: args.popular_authors_fraction,
                bigquery_table: args.popular_authors_bigquery_table.clone(),
                bigquery_key_path: args.popular_authors_bigquery_key_path.clone(),
                egress_proxy: args.popular_authors_egress_proxy.clone(),
                max_snapshot_age: Duration::from_secs(
                    args.popular_authors_max_snapshot_age_hours * 3600,
                ),
                once: args.popular_authors_job_once,
            },
        )
        .await;
    }
    if args.select_popular_posts {
        return select_popular_posts_from_stdin(selection_config(&args));
    }
    if args.popular_posts_job || args.popular_posts_job_once {
        let _stats =
            xai_stats_receiver::init_stats_receiver_guarded("home-mixer-popular-posts-job");
        let _tracing = xai_pipeline_tracing::init_tracing(
            "xai-home-mixer-popular-posts-job",
            &args.otel_endpoint,
        );
        xai_init_utils::init().rustls();
        return xai_home_mixer::popular_posts_job::run(
            xai_home_mixer::popular_posts_job::JobConfig {
                datacenter: args.datacenter.clone(),
                interval: Duration::from_secs(args.popular_posts_job_interval_secs),
                selection: selection_config(&args),
                once: args.popular_posts_job_once,
            },
        )
        .await;
    }
    let shard_coordinate = parse_shard(&args);

    xai_stringcenter::init_from_file(params::STRINGCENTER_BUNDLE_PATH);

    quality_factor::init(70.0, 30.0);

    let phoenix_xds = PhoenixXdsConfig {
        lb_policy: args.phoenix_xds_lb_policy,
        secondary_lb_policy: args.phoenix_xds_retrieval_lb_policy,
        lb_default_rtt_ms: args.phoenix_xds_lb_default_rtt_ms,
        h2_stream_window_bytes: args.phoenix_xds_h2_stream_window_bytes,
        h2_connection_window_bytes: args.phoenix_xds_h2_connection_window_bytes,
        socket_buffer_bytes: args.phoenix_xds_socket_buffer_bytes,
        aperture_size: args.phoenix_xds_aperture_size,
        health_probe: args.phoenix_xds_health_probe,
        grpc_health_timeout_ms: args.phoenix_xds_grpc_health_timeout_ms,
        grpc_health_interval_ms: args.phoenix_xds_grpc_health_interval_ms,
        readiness_port: args.phoenix_xds_readiness_port,
        readiness_path: args.phoenix_xds_readiness_path,
        discovery_authority: args.phoenix_xds_discovery_authority,
    };

    let vm_ranker_xds = VmRankerXdsConfig {
        lb_policy: args.vm_ranker_xds_lb_policy,
        secondary_lb_policy: "tonic".to_string(),
        lb_default_rtt_ms: args.vm_ranker_xds_lb_default_rtt_ms,
        h2_stream_window_bytes: args.vm_ranker_xds_h2_stream_window_bytes,
        h2_connection_window_bytes: args.vm_ranker_xds_h2_connection_window_bytes,
        socket_buffer_bytes: args.vm_ranker_xds_socket_buffer_bytes,
        aperture_size: args.vm_ranker_xds_aperture_size,
        health_probe: args.vm_ranker_xds_health_probe,
        grpc_health_timeout_ms: args.vm_ranker_xds_grpc_health_timeout_ms,
        grpc_health_interval_ms: args.vm_ranker_xds_grpc_health_interval_ms,
        readiness_port: args.vm_ranker_xds_readiness_port,
        readiness_path: args.vm_ranker_xds_readiness_path,
        discovery_authority: args.vm_ranker_xds_discovery_authority,
    };

    XServiceBuilder::new("home-mixer")
        .grpc_port(args.grpc_port)
        .metrics_port(args.metrics_port)
        .datacenter(args.datacenter)
        .otel_endpoint(args.otel_endpoint)
        .with_featureswitches_experiment_logging(params::FS_PATH)
        .with_fs_experiment_excluded_user_ids_param(params::FS_EXPERIMENT_EXCLUDED_USER_IDS_PARAM)
        .with_decider(params::decider_path(), None)
        .with_tls(TlsMode::server_mtls_from_env()?)
        .with_max_connection_age(Duration::from_secs(300))
        .with_reflection(pb::FILE_DESCRIPTOR_SET)
        .with_layer(dark_traffic_setup::resolve_layer())
        .with_layer(RejectDarkTrafficLayer::from_env())
        .http_routes(xai_profiling::profiling_router())
        .run::<HomeMixerServer>(HomeMixerConfig {
            shard_coordinate,
            phoenix_xds,
            vm_ranker_xds,
        })
        .await
}

async fn dump_popular_store(datacenter: &str, posts: bool) -> anyhow::Result<()> {
    use std::collections::HashMap;
    use xai_home_mixer::clients::popular_authors_store_client::{
        read_raw, POPULAR_AUTHORS_VERSION, POPULAR_POSTS_VERSION,
    };
    use xai_home_mixer::util::{popular_authors, popular_posts};

    let version = if posts {
        POPULAR_POSTS_VERSION
    } else {
        POPULAR_AUTHORS_VERSION
    };
    let Some(bytes) = read_raw(datacenter, version).await? else {
        println!("version={version} missing");
        return Ok(());
    };
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    if posts {
        let stored = popular_posts::decode_stored(&bytes).map_err(anyhow::Error::msg)?;
        let mut per_author: HashMap<u64, usize> = HashMap::new();
        for p in &stored.posts {
            *per_author.entry(p.author_id).or_default() += 1;
        }
        println!(
            "version={version} bytes={} generated_at_ms={} posts={} authors={} max_per_author={}",
            bytes.len(),
            stored.generated_at_ms,
            stored.posts.len(),
            per_author.len(),
            per_author.values().max().copied().unwrap_or(0)
        );
        for p in &stored.posts {
            println!("post {},{},{}", p.post_id, p.author_id, p.quality);
        }
    } else {
        let stored = popular_authors::decode_stored(&bytes).map_err(anyhow::Error::msg)?;
        println!(
            "version={version} bytes={} updated_at_ms={} authors={}",
            bytes.len(),
            stored.updated_at_ms,
            stored.authors.len()
        );
    }
    println!("hex={hex}");
    Ok(())
}

fn selection_config(args: &Args) -> xai_home_mixer::util::popular_posts::SelectionConfig {
    xai_home_mixer::util::popular_posts::SelectionConfig {
        per_author: args.popular_posts_per_author,
        budget: args.popular_posts_budget,
        half_life_hours: args.popular_posts_half_life_hours,
        ..Default::default()
    }
}

fn select_popular_posts_from_stdin(
    config: xai_home_mixer::util::popular_posts::SelectionConfig,
) -> anyhow::Result<()> {
    use std::io::Read;
    use xai_home_mixer::util::popular_posts::{select_popular_posts, PostViews};

    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text)?;
    let mut posts = Vec::new();
    for line in text
        .lines()
        .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
    {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        anyhow::ensure!(fields.len() >= 4, "bad row: {line}");
        posts.push(PostViews {
            post_id: fields[0].parse()?,
            author_id: fields[1].parse()?,
            age_hours: fields[2].parse()?,
            views: fields[3].parse()?,
        });
    }
    println!("post_id,author_id,quality");
    for p in select_popular_posts(&posts, &config) {
        println!("{},{},{}", p.post_id, p.author_id, p.quality);
    }
    Ok(())
}
