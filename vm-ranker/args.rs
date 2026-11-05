use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "VM Ranker — value-model ranking service for home timeline candidates", long_about = None)]
pub struct Args {
    #[arg(long, default_value_t = 9090)]
    pub grpc_port: u16,

    #[arg(long, default_value_t = 8080)]
    pub http_port: u16,

    #[arg(long, default_value_t = 256)]
    pub max_concurrent_requests: usize,

    #[arg(long, default_value_t = false)]
    pub dpp_enabled: bool,

    #[arg(long, default_value_t = 50)]
    pub dpp_top_k: usize,

    #[arg(long, default_value_t = 0.5)]
    pub dpp_theta: f64,

    #[arg(long, default_value_t = 100)]
    pub dpp_max_selected_rank: usize,

    #[arg(long, default_value_t = 1024)]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 0)]
    pub dpp_debug_viewer_id: u64,

    #[arg(long, default_value_t = false)]
    pub enable_profiling: bool,

    #[arg(long)]
    pub user_debias_artifact_path: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 300)]
    pub user_debias_reload_secs: u64,

    #[arg(long)]
    pub carousel_model_dir: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 60)]
    pub carousel_model_reload_secs: u64,

    #[arg(long, default_value_t = false)]
    pub config_sync_enabled: bool,

    #[arg(long, default_value = "https://config-git.twitter.biz/ro/config")]
    pub config_sync_remote: String,

    #[arg(long, default_value = "master")]
    pub config_sync_branch: String,

    #[arg(long, default_value = "/var/lib/vm-ranker/config")]
    pub config_sync_root: std::path::PathBuf,

    #[arg(long, default_value_t = 30)]
    pub config_sync_interval_secs: u64,

    #[arg(long, default_value_t = 300)]
    pub config_sync_initial_timeout_secs: u64,

    #[arg(long)]
    pub fs_impressions_datacenter: Option<String>,
}
