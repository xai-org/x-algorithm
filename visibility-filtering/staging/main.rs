#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

use clap::{Parser, Subcommand};
use xai_visibility_filtering_service::server::{self, ServeArgs};
use xai_visibility_filtering_service::staging::serve::StagingServer;

#[derive(Parser, Debug)]
#[command(about = "Visibility Filtering staging tools")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Serve {
        #[command(flatten)]
        serve: ServeArgs,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Serve { serve } => server::serve::<StagingServer>(serve).await,
    }
}
