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

    #[arg(
        long,
        default_value_t = 100,
        help = "Maximum DPP candidate pool size; requests may select a smaller pool"
    )]
    pub dpp_max_selected_rank: usize,

    #[arg(long, default_value_t = 1024)]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 0)]
    pub dpp_debug_viewer_id: u64,

    #[arg(long, default_value_t = false)]
    pub enable_profiling: bool,
}
